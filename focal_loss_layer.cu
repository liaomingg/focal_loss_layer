#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void FocalLossForwardGPU(const int nthreads,
				const Dtype * prob_data, const Dtype * label, Dtype* loss,
				const int num, const int dim, const int spatial_dim,
				const bool has_ignore_label_, const int ignore_label_,
				Dtype * counts, const Dtype alpha_, const Dtype gamma_) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int n = index / spatial_dim; 
			const int s = index % spatial_dim; 
			const int label_value = static_cast<int>(label[n * spatial_dim + s]);
			if (has_ignore_label_ && label_value == ignore_label_) {
				loss[index] = 0;
				counts[index] = 0;
			} else {
				const Dtype pk = max(prob_data[n * dim + label_value * spatial_dim + s], Dtype(FLT_MIN));
				loss[index] = -1 * alpha_ * powf(1 - pk, gamma_) * log(pk);
				counts[index] = 1;
			}
		}
	}
	
	template <typename Dtype>
	void FocalLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype> *> & bottom, const vector<Blob<Dtype> *> & top) {
		softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
		const Dtype * prob_data = prob_.gpu_data();
		const Dtype * label = bottom[1]->gpu_data();
		const int dim = prob_.count() / outer_num_;
		const int nthreads = outer_num_ * inner_num_;
		
		Dtype * loss_data = bottom[0]->mutable_gpu_diff();
		Dtype * counts = prob_.mutable_gpu_diff();
		FocalLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
			outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts, alpha_, gamma_);
		Dtype loss;
		caffe_gpu_asum(nthreads, loss_data, &loss);
		Dtype valid_count = -1;
		if (normalization_ == LossParameter_NormalizationMode_VALID &&
			has_ignore_label_) {
			caffe_gpu_asum(nthreads, counts, & valid_count);
		}
		top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
																valid_count);
		if (2 == top.size()) {
			top[1]->ShareData(prob_);
		}
	}
	
	template <typename Dtype>
	__global__ void FocalLossBackwardGPU(const int nthreads, const Dtype * prob_data,
				const Dtype * label, Dtype * bottom_diff, const int num, const int dim,
				const int spatial_dim, const bool has_ignore_label_,
				const int ignore_label_, Dtype * counts, const Dtype alpha_, const Dtype gamma_) {
		const int channels = dim / spatial_dim;
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int n = index / spatial_dim;
			const int s = index % spatial_dim;
			const int label_value = static_cast<int>(label[n * spatial_dim + s]);
			if (has_ignore_label_ && label_value == ignore_label_) {
				for (int c = 0; c < channels; ++c) {
					bottom_diff[n * dim + c * spatial_dim + s] = 0;
				}
				counts[index] = 0;
			} else {
				int c = 0;
				const Dtype pk = max(prob_data[n * dim + label_value * spatial_dim + s], Dtype(FLT_MIN));
				for (c = 0; c < label_value; ++c) {
					const Dtype pj = max(prob_data[n * dim + c * spatial_dim + s], Dtype(FLT_MIN));
					bottom_diff[n * dim + c * spatial_dim + s] = Dtype(
						-1 * alpha_ * (gamma_ * pow(1 - pk, gamma_ - 1) * pk * pj * log(pk) - pow(1 - pk, gamma_) * pj));
				}
				bottom_diff[n * dim + c * spatial_dim + s] = Dtype(
					-1 * alpha_ * (-1 * gamma_ * pow(1 - pk, gamma_) * pk * log(pk) + pow(1 - pk, gamma_ + 1)));
				c++;
				for ( ; c < channels; ++c) {
					const Dtype pj = max(prob_data[n * dim + c * spatial_dim + s], Dtype(FLT_MIN));
					bottom_diff[n * dim + c * spatial_dim + s] = Dtype(
						-1 * alpha_ * (gamma_ * pow(1 - pk, gamma_ - 1) * pk * pj * log(pk) - pow(1 - pk, gamma_) * pj));
				}
				counts[index] = 1;
			}
		}
	}
	
	template <typename Dtype>
	void FocalLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> & top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype> *> & bottom) {
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
						<< " Layer cannot backpropagate to label inputs.";
		}
		if (propagate_down[0]) {
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			const Dtype* prob_data = prob_.gpu_data();
			const Dtype* top_data = top[0]->gpu_data();
			const Dtype* label = bottom[1]->gpu_data();
			const int dim = prob_.count() / outer_num_;
			const int nthreads = outer_num_ * inner_num_;
			// Since this memory is nerver used for anything else,
			// we use to to avoid allocating new GPU memory 
			Dtype* counts = prob_.mutable_gpu_diff();
			// NOLINT_NEXT_LINE(whitespace/operators)
			FocalLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
				CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, bottom_diff,
				outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts,
				alpha_, gamma_);
			Dtype valid_count = -1;
			// Only launch another CUDA kernel if we actually need the count of valid
			// outputs.
			if (normalization_ == LossParameter_NormalizationMode_VALID &&
				has_ignore_label_) {
				caffe_gpu_asum(nthreads, counts, & valid_count);
			}
			const Dtype loss_weight = top[0]->cpu_diff()[0] / 
									get_normalizer(normalization_, valid_count);
			caffe_gpu_scal(prob_.count(), loss_weight, bottom_diff);
		}
	}
	
	INSTANTIATE_LAYER_GPU_FUNCS(FocalLossLayer);
} // namespace caffe
