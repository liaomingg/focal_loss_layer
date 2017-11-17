#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/focal_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void FocalLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);// softmax_bottom_vec_[0]指向bottom[0]
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);// soft_top_vec_[0]指向prob_
  // 建立softmax_layer, bottom[0] 即为该层输入， top[0]为prob_
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  	// 获取alpha 和 gamma参数
  	alpha_ = this->layer_param_.focal_loss_param().alpha();
	gamma_ = this->layer_param_.focal_loss_param().gamma();
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis()); // 指明在哪个轴上做softmax分类
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}
// 获取归一化因子
template <typename Dtype>
Dtype FocalLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer; // 声明一个归一化因子normalizer
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_); // label元素总个数
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);			// 归一化因子为有效元素的个数
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);				// 归一化因子为batch_size
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);						// 不指定归一化因子， 使用默认的归一化因子: 1
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}
// 前向传播 修改了
template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  // 先用softmax分类器来计算该点在各类别上的概率值prob属于[0, 1]
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_; // c * h * w
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
	  // 单点loss = -log(p) 修改loss需要修改这里
	  const Dtype pk = prob_data[i * dim + label_value * inner_num_ + j];
	  loss -= alpha_ * powf(1 - pk, gamma_) * log(std::max(pk, Dtype(FLT_MIN)));
      ++count; // 统计有效元素的个数
    }// 
  }// per_num
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}
// 反向传播 需要修改
template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// bottom[0]->cpu_data(), 即为zk
	// 对label进行反向传播
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  // 对bottom[0]进行反向传播
  if (propagate_down[0]) {
	// 获取bottom[0]的梯度数据指针
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	// 获取各点的概率值
    const Dtype* prob_data = prob_.cpu_data();
	// 如果i ！= k，则diff = prob
    //caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
	// 获取bottom[0]的cpu数据， 即zj
	
	// const Dtype* bottom_data = bottom[0]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
	int num_channel = bottom[0]->shape(softmax_axis_);
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
			  for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
				bottom_diff[i * dim + c * inner_num_ + j] = 0;
			  }
        } else {
			++count;
			int c = 0;
			const Dtype pk = std::max(prob_data[i * dim + label_value * inner_num_ + j], Dtype(FLT_MIN));
			for (c = 0; c < label_value; ++c) {
				const Dtype pj = std::max(prob_data[i * dim + c * inner_num_ + j], Dtype(FLT_MIN));
				bottom_diff[i * dim + c * inner_num_ + j] = Dtype(-1 * alpha_ * (gamma_ * pow(1 - pk, gamma_ - 1) * pk * pj * log(pk) - pow(1 - pk, gamma_) * pj)); // j != k
			    //bottom_diff[i * dim + c * inner_num_ + j] = Dtype (-1 * alpha_ * (gamma_ * powf(1 - pk, gamma_ - 1)* pk * pk * exp(bottom_data[i * dim + c * inner_num_ + j] - bottom_data[i * dim + label_value * inner_num_ + j]) * log(pk) - powf(1 - pk, gamma_) * pj) );// j != k
			} // per_channel
			bottom_diff[i * dim + label_value * inner_num_ + j] = Dtype (-1 * alpha_ * (-1 * gamma_ * powf(1 - pk, gamma_) * pk * log(pk) + powf(1 - pk, gamma_ + 1)));	// j = k
			c = c + 1;
			for (c; c < num_channel; ++c) {
				const Dtype pj = std::max(prob_data[i * dim + c * inner_num_ + j], Dtype(FLT_MIN));
				bottom_diff[i * dim + c * inner_num_ + j] = Dtype(-1 * alpha_ * (gamma_ * pow(1 - pk, gamma_ - 1) * pk * pj * log(pk) - pow(1 - pk, gamma_) * pj)); // j != k
				//bottom_diff[i * dim + c * inner_num_ + j] = Dtype(-1 * alpha_ * (gamma_ * powf(1 - pk, gamma_ - 1) * pk * pk * exp(bottom_data[i * dim + c * inner_num_ + j] - bottom_data[i * dim + label_value * inner_num_ + j]) * log(pk) - powf(1 - pk, gamma_) * pj));// j != k
			} // per_channel
        }
      }// per_h_w
    }// per_num
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(FocalLossLayer);
#endif

INSTANTIATE_CLASS(FocalLossLayer);
REGISTER_LAYER_CLASS(FocalLoss);

}  // namespace caffe
