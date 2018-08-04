#include <dlpack/dlpack.h>
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>
#include <tvm/operation.h>
#include <tvm/build_module.h>
// new namespoace
namespace test {
// register int vector as extension type
using FloatVector = std::vector<float>;
}  // namespace test

namespace tvm {
namespace runtime {

template<>
struct extension_class_info<test::FloatVector> {
  static const int code = kExtBegin + 1;
};
}  // runtime
}  // tvm

// do registration, this need to be in cc file
TVM_REGISTER_EXT_TYPE(test::FloatVector);

TEST(BuildModule, Basic) {
  using namespace tvm;
  auto n = var("n");
  Array<Expr> shape;
  shape.push_back(n);

  auto A = placeholder(shape, Float(32), "A");
  auto B = placeholder(shape, Float(32), "B");

  auto C = compute(A->shape, [&A, &B](Expr i) {
    return A[i] + B[i];
  }, "C");

  auto s = create_schedule({ C->op });

  auto cAxis = C->op.as<ComputeOpNode>()->axis;

  IterVar bx, tx;
  s[C].split(cAxis[0], 64, &bx, &tx);

  auto args = Array<Tensor>({ A, B, C });
  std::unordered_map<Tensor, Buffer> binds;

  auto config = build_config();
  auto target = target::llvm();

  auto lowered = lower(s, args, "func", binds, config);
  auto module = build(lowered, target, Target(), config);
//  std::cout << module->type_key() << std::endl;
//  std::cout << module->GetSource() << std::endl;

  auto func = module->GetFunction("func", false);

  {
      DLTensor* a;
      DLTensor* b;
      DLTensor* c;
      int ndim = 1;
      int dtype_code = kDLFloat;
      int dtype_bits = 32;
      int dtype_lanes = 1;
      int device_type = kDLCPU;
      int device_id = 0;
      int64_t dlshape[] = {10};
      TVMArrayAlloc((tvm_index_t*)dlshape, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &a);
      TVMArrayAlloc((tvm_index_t*)dlshape, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &b);
      TVMArrayAlloc((tvm_index_t*)dlshape, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &c);
      for (int i = 0; i < dlshape[0]; ++i) {
        static_cast<float*>(a->data)[i] = 1;
        static_cast<float*>(b->data)[i] = 2;
        static_cast<float*>(c->data)[i] = 0;
      }
      func(a, b, c);
      for (int i = 0; i < dlshape[0]; ++i) {
        std::cout << static_cast<float*>(c->data)[i] << std::endl;
      }
  }


#if 0
  auto mali_target = Target::create("opencl -model=Mali-T860MP4@800Mhz -device=mali");

  CHECK_EQ(mali_target->str(), "opencl -model=Mali-T860MP4@800Mhz -device=mali");
#endif
}


int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
