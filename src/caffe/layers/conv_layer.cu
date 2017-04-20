#include <vector>

#include "caffe/layers/conv_layer.hpp"

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
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight;
  // DNS
  if (sparsity_term_ && this->phase_ == TRAIN) {
      const Dtype* const_weight_mask;
      Dtype* weight_mask;
      Dtype* weight_mask_diff;
      const Dtype* const_weight_mask_diff;
      const int weight_number = this->blobs_[0]->count();
      if (this->bias_term_) {
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
        
        // According to 68-95-99.7 rule, we prune the weights distributed between [-0.68*Delte, 0.68*Delte].
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
        if (this->bias_term_) {
          const_weight_mask = this->blobs_[2]->gpu_data();
        } else {
          const_weight_mask = this->blobs_[1]->gpu_data();
        }
        caffe_gpu_mul(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), 
                        const_weight_mask, this->blobs_[0]->mutable_gpu_data());
      }
      weight = this->blobs_[0]->gpu_data();
  }
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight; 
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  if (sparsity_term_) {
      if (this->bias_term_) {
          weight = this->blobs_[2]->gpu_diff();
      } else {
          weight = this->blobs_[1]->gpu_diff();
      }
  } else {
      weight = this->blobs_[0]->gpu_data();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
