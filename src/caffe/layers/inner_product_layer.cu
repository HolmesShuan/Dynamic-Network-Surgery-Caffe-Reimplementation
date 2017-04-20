#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaskGenerator(const int n, const Dtype* weight, Dtype* weight_mask, 
        const Dtype upperlimit, const Dtype lowerlimit) {
  CUDA_KERNEL_LOOP(index, n) {
    weight_mask[index] = weight[index] > lowerlimit ? 
            (weight[index] < upperlimit ? (Dtype)0. : (Dtype)1.) : (Dtype)1.;
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight;
  // DNS
  if (sparsity_term_ && this->phase_ == TRAIN) {
      const Dtype* const_weight_mask;
      Dtype* weight_mask;
      Dtype* weight_mask_diff;
      const Dtype* const_weight_mask_diff;
      const int weight_number = this->blobs_[0]->count();
      if (bias_term_) {
          weight_mask = this->blobs_[2]->mutable_gpu_data();
          const_weight_mask = this->blobs_[2]->gpu_data();
          weight_mask_diff = this->blobs_[2]->mutable_gpu_diff();
          const_weight_mask_diff = this->blobs_[2]->gpu_diff();
      } else {
          weight_mask = this->blobs_[1]->mutable_gpu_data();
          const_weight_mask = this->blobs_[1]->gpu_data();
          weight_mask_diff = this->blobs_[1]->mutable_gpu_diff();
          const_weight_mask_diff = this->blobs_[1]->gpu_diff();
      }
      if (this->surgey_term_) {
        Dtype mean_value;
        Dtype std_value;
        caffe_gpu_set(weight_number, (Dtype)1., weight_mask_diff);
        caffe_gpu_dot(weight_number, const_weight_mask_diff, this->blobs_[0]->gpu_data(), &mean_value);
        mean_value /= Dtype(weight_number);
        caffe_gpu_scalar(weight_number, -mean_value, this->blobs_[0]->gpu_data(), weight_mask_diff);
        caffe_gpu_mul(weight_number, const_weight_mask_diff, const_weight_mask_diff, weight_mask_diff);
        caffe_gpu_asum(weight_number, const_weight_mask_diff, &std_value);
        std_value /= Dtype(weight_number);
        std_value = sqrt(std_value);
        
        // According to 68-95-99.7 rule, we prune the weights distributed between [-1.281*Delte, 1.281*Delte].
        // More Details: https://en.wikipedia.org/wiki/Standard_deviation
        // We tried to make the selecion of hyperparameter more convencing.
        
        const Dtype upper_threshold_value = mean_value + threshold*std_value;
        const Dtype lower_threshold_value = mean_value - threshold*std_value;
        
        MaskGenerator<Dtype><<<CAFFE_GET_BLOCKS(weight_number), CAFFE_CUDA_NUM_THREADS>>>(weight_number, 
            this->blobs_[0]->gpu_data(), weight_mask, upper_threshold_value, lower_threshold_value);
        
        Dtype s_connection_num;
        Dtype a_connection_num;
        caffe_gpu_asum(weight_number, const_weight_mask, &a_connection_num);
        s_connection_num = weight_number - a_connection_num;
        this->s_connection_num = s_connection_num;
        this->a_connection_num = a_connection_num;
      }
      caffe_gpu_mul(weight_number, this->blobs_[0]->gpu_data(), const_weight_mask, weight_mask_diff);
      weight = const_weight_mask_diff;
  } else {
      if (sparsity_term_) {
        const Dtype* const_weight_mask;
        if (bias_term_) {
          const_weight_mask = this->blobs_[2]->gpu_data();
        } else {
          const_weight_mask = this->blobs_[1]->gpu_data();
        }
        caffe_gpu_mul(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), 
                        const_weight_mask, this->blobs_[0]->mutable_gpu_data());
      }
      weight = this->blobs_[0]->gpu_data();
  }
  if (M_ == 1) { 
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* weight;
    if (sparsity_term_) {
      if (this->bias_term_) {
          weight = this->blobs_[2]->gpu_diff();
      } else {
          weight = this->blobs_[1]->gpu_diff();
      }
  } else {
      weight = this->blobs_[0]->gpu_data();
  }
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, weight,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, weight,
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
