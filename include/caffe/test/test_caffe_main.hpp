// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "caffe/common.hpp"

using std::cout;
using std::endl;

#ifdef CMAKE_BUILD
  #include "caffe_config.h"
#else
  #define CUDA_TEST_DEVICE -1
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define ABS_TEST_DATA_DIR "src/caffe/test/test_data"
#endif

int main(int argc, char** argv);

namespace caffe {

template <typename TypeParam>
class MultiDeviceTest : public ::testing::Test {
 public:
  typedef typename TypeParam::Dtype Dtype;
 protected:
  MultiDeviceTest() {
    Caffe::set_mode(TypeParam::device);
  }
  virtual ~MultiDeviceTest() {}
};

// removed double Dtype because it crashes DecodeBBoxesKernel in bbox_util.cu
// and is not needed for training with float
typedef ::testing::Types<float> TestDtypes;
//typedef ::testing::Types<float, double> TestDtypes;

  template <typename TypeParam>
struct CPUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::CPU;
};

template <typename Dtype>
class CPUDeviceTest : public MultiDeviceTest<CPUDevice<Dtype> > {
};

#ifdef CPU_ONLY

typedef ::testing::Types<CPUDevice<float>,
                         CPUDevice<double> > TestDtypesAndDevices;

#else

template <typename TypeParam>
struct GPUDevice {
  typedef TypeParam Dtype;
  static const Caffe::Brew device = Caffe::GPU;
};

template <typename Dtype>
class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype> > {
};

// removed GPUDevice<double> because it crashes DecodeBBoxesKernel in bbox_util.cu
// and is not needed for training with float
typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
                         GPUDevice<float> >
                         TestDtypesAndDevices;
//typedef ::testing::Types<CPUDevice<float>, CPUDevice<double>,
//                         GPUDevice<float>, GPUDevice<double> >
//                         TestDtypesAndDevices;
#endif

}  // namespace caffe

#endif  // CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
