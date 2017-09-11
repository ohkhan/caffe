#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  // reverted back to copying instead of forwarding the pointer.
  // for some reason this breaks the multibox loss layer test
//  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  caffe_copy(prefetch_current_->data_.count(), prefetch_current_->data_.gpu_data(),
             top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    // reverted back to copying instead of forwarding the pointer.
    // for some reason this breaks the multibox loss layer test
//    top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
    caffe_copy(prefetch_current_->label_.count(), prefetch_current_->label_.gpu_data(),
               top[1]->mutable_gpu_data());
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
