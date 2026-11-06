"""桥接契约测试：编译真实桥接源码，使用明确标记的 RKNN 测试替身。

替身只验证 ABI、返回值和资源生命周期；这里的张量/耗时不是 NPU 测量。
板端执行：python tests/test_bridge.py -v。无需 pytest 或新增依赖。
"""
from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from typing import Any, ClassVar
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# 按 vendor/rknn_api.h 编译，禁止在 Python 猜测厂商结构体布局。
FAKE_RUNTIME = r'''
#include "rknn_api.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace {
int fail_stage = 0, geometry = 0;
int calls[8] = {};
uint32_t flags_seen = 0;
int mask_seen = 0;
bool live = false;
std::array<std::vector<float>, 9> storage;
char report[] = "TEST_ONLY_OPERATOR_REPORT\n";
void attribute(rknn_tensor_attr* a, bool output) {
    uint32_t index = a->index;
    *a = {};
    a->index = index;
    a->n_dims = 4;
    a->fmt = RKNN_TENSOR_NCHW;
    a->type = RKNN_TENSOR_FLOAT16;
    a->qnt_type = RKNN_TENSOR_QNT_NONE;
    a->dims[0] = 1;
    if (output) {
        const uint32_t channels[] = {64, 80, 1};
        a->dims[1] = channels[index % 3];
        a->dims[2] = a->dims[3] = 80u >> (index / 3);
        std::strcpy(a->name, "test\"output\\name\n");
    } else {
        a->dims[1] = 3;
        a->dims[2] = a->dims[3] = geometry == 4 ? 320 : 640;
        std::strcpy(a->name, "test_input");
    }
    a->n_elems = a->dims[0] * a->dims[1] * a->dims[2] * a->dims[3];
    a->size = a->n_elems * 2;
    a->size_with_stride = a->size;
    if (geometry == 2 && output) a->n_elems += 1;
    if (geometry == 5 && output) a->n_dims = RKNN_MAX_DIMS + 1;
}
}
extern "C" {
void fake_reset() {
    fail_stage = geometry = mask_seen = 0;
    flags_seen = 0;
    live = false;
    std::fill(calls, calls + 8, 0);
}
void fake_fail(int stage) { fail_stage = stage; }
void fake_geometry(int value) { geometry = value; }
int fake_count(int index) { return calls[index]; }
int fake_flags() { return flags_seen; }
int fake_mask() { return mask_seen; }
int rknn_init(rknn_context* ctx, void*, uint32_t size, uint32_t flags, rknn_init_extend* ext) {
    ++calls[0]; flags_seen = flags;
    if (fail_stage == 1) return -6;
    if (size != 0 || ext || (flags != 0 && flags != 8)) return -80;
    *ctx = 1;
    return 0;
}
int rknn_destroy(rknn_context) {
    ++calls[1];
    return fail_stage == 14 ? -14 : 0;
}
int rknn_set_core_mask(rknn_context, rknn_core_mask mask) {
    mask_seen = mask;
    return fail_stage == 2 ? -2 : 0;
}
int rknn_query(rknn_context, rknn_query_cmd cmd, void* ptr, uint32_t size) {
    switch (cmd) {
    case RKNN_QUERY_IN_OUT_NUM: {
        if (fail_stage == 3) return -3;
        if (size != sizeof(rknn_input_output_num)) return -81;
        auto* n = static_cast<rknn_input_output_num*>(ptr);
        n->n_input = 1; n->n_output = geometry == 1 ? 8 : 9;
        return 0;
    }
    case RKNN_QUERY_INPUT_ATTR:
    case RKNN_QUERY_OUTPUT_ATTR: {
        bool output = cmd == RKNN_QUERY_OUTPUT_ATTR;
        if (fail_stage == (output ? 5 : 4)) return -fail_stage;
        if (size != sizeof(rknn_tensor_attr)) return -82;
        attribute(static_cast<rknn_tensor_attr*>(ptr), output);
        return 0;
    }
    case RKNN_QUERY_SDK_VERSION: {
        if (fail_stage == 6) return -6;
        if (size != sizeof(rknn_sdk_version)) return -83;
        auto* v = static_cast<rknn_sdk_version*>(ptr);
        std::strcpy(v->api_version, "TEST_ONLY_SDK");
        std::strcpy(v->drv_version, "TEST_ONLY_DRIVER");
        return 0;
    }
    case RKNN_QUERY_MEM_SIZE: {
        if (fail_stage == 7) return -7;
        if (size != sizeof(rknn_mem_size)) return -84;
        auto* m = static_cast<rknn_mem_size*>(ptr);
        m->total_weight_size = 123;
        m->total_internal_size = 456;
        m->total_dma_allocated_size = (uint64_t(1) << 33) + 789;
        return 0;
    }
    case RKNN_QUERY_PERF_RUN: {
        ++calls[7];
        if (fail_stage == 12) return -12;
        if (!live || size != sizeof(rknn_perf_run)) return -85;
        static_cast<rknn_perf_run*>(ptr)->run_duration = 1234;
        report[0] = '\0';  // 后续 API 可使报告指针失效，桥接必须立即保存。
        return 0;
    }
    case RKNN_QUERY_PERF_DETAIL: {
        ++calls[6];
        if (fail_stage == 13) return -13;
        if (!live || flags_seen != 8 || size != sizeof(rknn_perf_detail)) return -86;
        std::strcpy(report, "TEST_ONLY_OPERATOR_REPORT\n");
        auto* p = static_cast<rknn_perf_detail*>(ptr);
        p->perf_data = report; p->data_len = std::strlen(report);
        return 0;
    }
    default: return -87;
    }
}
int rknn_inputs_set(rknn_context, uint32_t n, rknn_input* inputs) {
    ++calls[2];
    if (fail_stage == 8) return -8;
    if (n != 1 || !inputs[0].buf || inputs[0].index != 0 ||
        inputs[0].size != 640 * 640 * 3 || inputs[0].pass_through != 0 ||
        inputs[0].type != RKNN_TENSOR_UINT8 || inputs[0].fmt != RKNN_TENSOR_NHWC)
        return -88;
    return 0;
}
int rknn_run(rknn_context, rknn_run_extend* ext) {
    ++calls[3];
    if (fail_stage == 9) return -9;
    return ext || live ? -89 : 0;
}
int rknn_outputs_get(rknn_context, uint32_t n, rknn_output* outputs, rknn_output_extend* ext) {
    ++calls[4];
    if (fail_stage == 10) return -10;
    if (n != 9 || live || ext) return -90;
    live = true;
    for (uint32_t i = 0; i < n; ++i) {
        if (outputs[i].index != i || outputs[i].want_float != 1 || outputs[i].is_prealloc)
            return -91;
        rknn_tensor_attr attr{}; attr.index = i; attribute(&attr, true);
        storage[i].assign(attr.n_elems, float(100 * calls[3] + i));
        outputs[i].buf = storage[i].data();
        outputs[i].size = uint32_t(storage[i].size() * sizeof(float));
        if (geometry == 3 && i == 0) outputs[i].size -= sizeof(float);
    }
    return 0;
}
int rknn_outputs_release(rknn_context, uint32_t n, rknn_output* outputs) {
    ++calls[5];
    if (!live || n != 9) return -92;
    live = false;
    for (uint32_t i = 0; i < n; ++i) {
        std::fill(storage[i].begin(), storage[i].end(), -999.0f);
        outputs[i].buf = nullptr;
    }
    return fail_stage == 11 ? -11 : 0;
}
}
'''


class BridgeContract(unittest.TestCase):
    Session: ClassVar[Any] = None
    library: ClassVar[Path]
    native: ClassVar[Any]

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="rknn-bridge-contract-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.model = cls.directory / "fixture.rknn"
        cls.model.write_bytes(b"TEST_ONLY_NOT_A_MODEL")

    def load_bridge(self):
        self.assertTrue((ROOT / "scripts/rknn_bridge.cpp").is_file(),
                        "缺少 C++ 桥接实现")
        self.assertTrue((ROOT / "scripts/rknn_bridge.py").is_file(),
                        "缺少 Python 包装实现")
        if type(self).Session is None:
            self.assertIsNotNone(shutil.which("g++"), "契约测试需要已有 g++")
            shim = self.directory / "test_only_rknn.cpp"
            shim.write_text(FAKE_RUNTIME, encoding="utf-8")
            library = self.directory / "libbridge_contract.so"
            result = subprocess.run([
                "g++", "-std=c++17", "-shared", "-fPIC", "-O1", "-g",
                "-Wall", "-Wextra", "-Werror", "-Wl,-z,defs",
                "-I", str(ROOT / "vendor"),
                str(ROOT / "scripts/rknn_bridge.cpp"), str(shim),
                "-o", str(library),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            spec = importlib.util.spec_from_file_location(
                "rknn_bridge_under_test", ROOT / "scripts/rknn_bridge.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            type(self).Session = module.RknnSession
            type(self).library = library
            type(self).native = ctypes.CDLL(str(library))
        return type(self).Session

    def test_production_build_links_runtime_and_invalid_path_raises(self):
        Session = self.load_bridge()
        script = ROOT / "scripts/build_bridge.sh"
        self.assertTrue(script.is_file(), "缺少实际编译脚本")
        library = self.directory / "librknn_bridge_real.so"
        built = subprocess.run(["bash", str(script), str(library)], capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        with self.assertRaisesRegex(RuntimeError, "rknn_init.*-"):
            Session(self.directory / "does_not_exist.rknn", library)

    def test_rejects_wrong_model_geometry_at_initialization(self):
        Session = self.load_bridge()
        for geometry in (1, 2, 4, 5):
            with self.subTest(geometry=geometry):
                self.native.fake_reset()
                self.native.fake_geometry(geometry)
                session = None
                try:
                    with self.assertRaisesRegex(RuntimeError, "output|input|dims|YOLO"):
                        session = Session(self.model, self.library)
                finally:
                    if session is not None:
                        session.close()
                self.assertEqual(self.native.fake_count(1), 1)

    def test_rejects_invalid_input_without_submitting_to_rknn(self):
        Session = self.load_bridge()
        self.native.fake_reset()
        session = Session(self.model, self.library)
        rgb = np.full((640, 640, 3), 114, np.uint8)
        try:
            for bad in (rgb.astype(np.float32), rgb.transpose(0, 2, 1), rgb[:, ::-1], None):
                with self.subTest(input_type=type(bad).__name__):
                    with self.assertRaises((ValueError, TypeError)):
                        session.run(bad)
            self.assertEqual(self.native.fake_count(2), 0)
            with self.assertRaisesRegex(ValueError, "perf_path"):
                session.run(rgb, self.directory / "must_not_create.txt")
        finally:
            session.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.run(rgb)

    def test_every_initialization_api_error_destroys_context(self):
        Session = self.load_bridge()
        for stage in range(2, 8):
            with self.subTest(stage=stage):
                self.native.fake_reset()
                self.native.fake_fail(stage)
                with self.assertRaisesRegex(RuntimeError, f"rknn_.*-{stage}"):
                    Session(self.model, self.library)
                self.assertEqual(self.native.fake_count(1), 1)

    def test_every_run_api_error_is_raised_and_outputs_are_released(self):
        Session = self.load_bridge()
        rgb = np.full((640, 640, 3), 114, np.uint8)
        for stage in (8, 9, 10, 11, 12, 13):
            with self.subTest(stage=stage):
                self.native.fake_reset()
                session = Session(self.model, self.library, detail=stage == 13)
                self.native.fake_fail(stage)
                try:
                    with self.assertRaisesRegex(RuntimeError, f"rknn_.*-{stage}"):
                        session.run(rgb, self.directory / "error.perf.txt" if stage == 13 else None)
                    self.assertEqual(self.native.fake_count(5), int(stage in (11, 12, 13)))
                finally:
                    session.close()

    def test_output_size_mismatch_is_rejected_and_released(self):
        Session = self.load_bridge()
        self.native.fake_reset()
        session = Session(self.model, self.library)
        self.native.fake_geometry(3)
        try:
            with self.assertRaisesRegex(RuntimeError, "output buffer size"):
                session.run(np.full((640, 640, 3), 114, np.uint8))
            self.assertEqual(self.native.fake_count(5), 1)
        finally:
            session.close()

    def test_python_copy_failure_still_releases_outputs(self):
        Session = self.load_bridge()
        self.native.fake_reset()
        session = Session(self.model, self.library)
        try:
            with mock.patch("numpy.ctypeslib.as_array", side_effect=MemoryError("test copy failure")):
                with self.assertRaisesRegex(MemoryError, "test copy failure"):
                    session.run(np.full((640, 640, 3), 114, np.uint8))
            self.assertEqual(self.native.fake_count(5), 1)
        finally:
            session.close()

    def test_close_reports_destroy_error_and_remains_idempotent(self):
        Session = self.load_bridge()
        self.native.fake_reset()
        session = Session(self.model, self.library)
        self.native.fake_fail(14)
        with self.assertRaisesRegex(RuntimeError, "rknn_destroy.*-14"):
            session.close()
        session.close()
        self.assertEqual(self.native.fake_count(1), 1)

    def test_detail_requires_report_destination_and_save_failures_release(self):
        Session = self.load_bridge()
        self.native.fake_reset()
        session = Session(self.model, self.library, detail=True)
        rgb = np.full((640, 640, 3), 114, np.uint8)
        try:
            with self.assertRaisesRegex(ValueError, "perf_path"):
                session.run(rgb)
            self.assertEqual(self.native.fake_count(2), 0)
            with self.assertRaisesRegex(RuntimeError, "PERF_DETAIL path"):
                session.run(rgb, self.directory / "missing" / "detail.txt")
            self.assertEqual(self.native.fake_count(5), 1)
        finally:
            session.close()

    def test_context_manager_closes_context(self):
        Session = self.load_bridge()
        self.native.fake_reset()
        self.assertTrue(hasattr(Session, "__enter__"), "缺少上下文管理")
        with Session(self.model, self.library) as session:
            self.assertIn("outputs", session.metadata)
        self.assertEqual(self.native.fake_count(1), 1)

    def test_detail_saves_report_before_next_rknn_api(self):
        Session = self.load_bridge()
        self.native.fake_reset()
        session = Session(self.model, self.library, detail=True)
        report_path = self.directory / "detail.perf.txt"
        try:
            self.assertEqual(self.native.fake_flags(), 8)
            self.assertTrue(session.metadata["detail"])
            try:
                outputs, _ = session.run(np.full((640, 640, 3), 114, np.uint8), report_path)
            except RuntimeError as error:
                self.fail(f"详细模式应成功保存报告：{error}")
            self.assertEqual(len(outputs), 9)
            self.assertEqual(report_path.read_bytes(), b"TEST_ONLY_OPERATOR_REPORT\n")
            self.assertEqual(self.native.fake_count(6), 1)
            self.assertEqual(self.native.fake_count(5), 1)
        finally:
            session.close()

    def test_resident_run_copies_all_outputs_before_release(self):
        Session = self.load_bridge()
        self.native.fake_reset()
        session = Session(self.model, self.library)
        try:
            self.assertTrue(callable(getattr(session, "run", None)), "缺少 run 接口")
            rgb = np.full((640, 640, 3), 114, dtype=np.uint8)
            first, timing = session.run(rgb)
            second, _ = session.run(rgb)
            self.assertEqual(len(first), 9)
            for i, (a, b) in enumerate(zip(first, second)):
                self.assertEqual(a.shape, tuple(session.metadata["outputs"][i]["dims"]))
                self.assertEqual(a.dtype, np.float32)
                self.assertTrue(np.all(a == 100 + i), "释放/下次推理不能改变先前输出")
                self.assertTrue(np.all(b == 200 + i))
                self.assertFalse(np.shares_memory(a, b))
            self.assertEqual(self.native.fake_count(0), 1, "模型必须常驻")
            self.assertEqual(self.native.fake_count(5), 2, "每次复制后都必须释放")
            self.assertEqual(self.native.fake_count(6), 0, "baseline 不得查询 PERF_DETAIL")
            self.assertEqual(timing["perf_run_us"], 1234)
            for key in ("inputs_set_ms", "run_ms", "outputs_get_ms", "output_copy_ms",
                        "outputs_release_ms", "perf_query_ms", "wall_ms"):
                self.assertGreaterEqual(timing[key], 0, key)
            self.assertGreaterEqual(timing["wall_ms"], timing["output_copy_ms"])
        finally:
            session.close()

    def test_invalid_model_raises_with_api_and_status(self):
        Session = self.load_bridge()
        self.native.fake_reset()
        self.native.fake_fail(1)
        with self.assertRaisesRegex(RuntimeError, r"rknn_init.*-6"):
            Session(self.model, self.library)

    def test_metadata_covers_every_tensor_and_context_is_closed_once(self):
        Session = self.load_bridge()
        self.native.fake_reset()
        session = Session(self.model, self.library)
        try:
            self.assertTrue(hasattr(session, "metadata"), "缺少实际查询元数据")
            meta = session.metadata
            self.assertEqual(meta["n_input"], 1)
            self.assertEqual(meta["n_output"], 9)
            self.assertEqual(len(meta["inputs"]), 1)
            self.assertEqual(len(meta["outputs"]), 9)
            self.assertEqual(meta["inputs"][0]["dims"], [1, 3, 640, 640])
            for index, attr in enumerate(meta["outputs"]):
                self.assertEqual(attr["index"], index)
                self.assertEqual(attr["name"], 'test"output\\name\n')
                self.assertEqual(attr["fmt"], "NCHW")
                self.assertEqual(attr["type"], "FP16")
                self.assertIn("size_with_stride", attr)
            self.assertEqual(meta["sdk_version"]["api_version"], "TEST_ONLY_SDK")
            self.assertEqual(meta["sdk_version"]["drv_version"], "TEST_ONLY_DRIVER")
            self.assertEqual(meta["mem_size"]["total_dma_allocated_size"], 2**33 + 789)
            self.assertEqual(meta["core_mask"], 7)
            self.assertEqual(self.native.fake_mask(), 7)
            self.assertEqual(meta["init_flags"], 0)
            self.assertGreaterEqual(meta["init_ms"], 0)
            self.assertGreaterEqual(meta["create_ms"], meta["init_ms"])
            self.assertEqual(self.native.fake_count(0), 1)
        finally:
            session.close()
            session.close()
        self.assertEqual(self.native.fake_count(1), 1)


if __name__ == "__main__":
    unittest.main()
