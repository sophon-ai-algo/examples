#include "yolox.hpp"
#include "utils.hpp"

using namespace std;

YOLOX::YOLOX(bm_handle_t& bm_handle, const std::string bmodel, vector<int> strides)
    : p_bmrt_(NULL)
{
    bm_handle_ = bm_handle;
    // init bmruntime contxt
    p_bmrt_ = bmrt_create(bm_handle_);
    if (NULL == p_bmrt_) {
        cout << "ERROR: get handle failed!" << endl;
        exit(1);
    }
      // load bmodel from file
    bool ret = bmrt_load_bmodel(p_bmrt_, bmodel.c_str());
    if (!ret) {
        cout << "ERROR: Load bmodel[" << bmodel << "] failed" << endl;
        exit(1);
    }

    const char **net_names;
    bmrt_get_network_names(p_bmrt_, &net_names);
    int net_count = bmrt_get_network_number(p_bmrt_);
    for (size_t i = 0; i < net_count; i++)
    {
        printf("net_names[%d]: %s\n",i, net_names[i]);
    }

    // get model info by model name
    net_name_ = std::string(net_names[0]);
    net_info_ = bmrt_get_network_info(p_bmrt_, net_names[0]);
    if (NULL == net_info_) {
        cout << "ERROR: get net-info failed!" << endl;
        exit(1);
    }
    free(net_names);

    // get data type
    if (NULL == net_info_->input_dtypes) {
        cout << "ERROR: get net input type failed!" << endl;
        exit(1);
    }
    // get data type
    if (NULL == net_info_->input_dtypes) {
        cout << "ERROR: get net input type failed!" << endl;
        exit(1);
    }
    if (BM_FLOAT32 == net_info_->input_dtypes[0]) {
        threshold_ = 0.6;
        nms_threshold_ = 0.45;
        is_int8_ = false;
        printf("input data type: BM_FLOAT32!\n");
    } else {
        threshold_ = 0.52;
        nms_threshold_ = 0.45;
        is_int8_ = true;
        printf("input data type: BM_INT8!\n");
    }

    if (BM_FLOAT32 == net_info_->output_dtypes[0]) {
        output_scale_ = net_info_->output_scales[0];
        output_is_int8_ = false;
        printf("output data type: BM_FLOAT32!,output scale is: %f\n",output_scale_);
    } else {
        output_scale_ = net_info_->output_scales[0];
        output_is_int8_ = true;
        printf("output data type: BM_INT8!,output scale is: %f\n",output_scale_);
    }

    int input_num = net_info_->input_num;
    // for (size_t i = 0; i < input_num; i++)    {
    //     printf("input[%d]: %s\n",i,net_info_->input_names[i]);
    // }

    int output_num = net_info_->output_num;
    // for (size_t i = 0; i < output_num; i++)    {
    //     printf("output[%d]: %s\n",i,net_info_->output_names[i]);
    // }

    int stage_num = net_info_->stage_num;
    // printf("stage_num: %d\n",stage_num);
    input_shape_ = net_info_->stages[0].input_shapes[0];
    output_shape_ = net_info_->stages[0].output_shapes[0];


    // printf("output_shape_: %d\n",output_shape_.dims[1]);
    
    // printf("Input Dim: %d -- [ ",input_shape_.num_dims);
    // for (size_t i=0; i<input_shape_.num_dims;i++){
    //     printf("%d ",input_shape_.dims[i]);
    // }
    // printf("]\n");

    input_width = input_shape_.dims[3];
    input_height = input_shape_.dims[2];

    outlen_dim1_ = 0;
    for(int i=0;i<strides.size();i++){
        int layer_w = input_width/strides[i];
        int layer_h = input_height/strides[i];
        outlen_dim1_ += layer_w * layer_w;
    }
    if(outlen_dim1_ != output_shape_.dims[1] ){
        cout << "ERROR: strides set failed!" << endl;
        exit(1);
    }
    outlen_dim1_ =  output_shape_.dims[1];
    classes_ = output_shape_.dims[2] - 5;
    channels_resu_ = output_shape_.dims[2];
    grids_x_ = new int[outlen_dim1_];
    grids_y_ = new int[outlen_dim1_];
    expanded_strides_ = new int[outlen_dim1_];

    int channel_len = 0;
    for(int i=0;i<strides.size();i++){
        int layer_w = input_width/strides[i];
        int layer_h = input_height/strides[i];
        // printf("layer_w: %d, layer_h: %d\n",layer_w,layer_h);
        for(int m=0;m<layer_h;++m){
            for(int n=0;n<layer_w;++n){
                grids_x_[channel_len+m*layer_h+n] = n;
                grids_y_[channel_len+m*layer_h+n] = m;
                expanded_strides_[channel_len+m*layer_h+n] = strides[i];
            }
        }
        channel_len += layer_w * layer_h;
    }
    max_batch_ = output_shape_.dims[0];
    output_size = 1;
    // printf("Output Dim: %d -- [ ",output_shape_.num_dims);
    for (size_t i=0; i<output_shape_.num_dims;i++){
        // printf("%d ",output_shape_.dims[i]);
        output_size = output_size * output_shape_.dims[i];
    }
    // printf("]\n");

    resize_bmcv_ = new bm_image[max_batch_];
    linear_trans_bmcv_ = new bm_image[max_batch_];
    output_ = new float[output_size];
    int output_count = bmrt_shape_count(&output_shape_);
    // printf("output_size: %d, output_count: %d\n",output_size,output_count);
      // init bm images for storing results of combined operation of resize & crop & split
    bm_status_t bm_ret = bm_image_create_batch(bm_handle_,
                                input_height,
                                input_width,
                                FORMAT_BGR_PLANAR,
                                DATA_TYPE_EXT_1N_BYTE,
                                resize_bmcv_,
                                max_batch_);

    if (BM_SUCCESS != bm_ret) {
        cout << "ERROR: bm_image_create_batch failed" << endl;
        exit(1);
    }

    // bm images for storing inference inputs
    bm_image_data_format_ext data_type;
    if (is_int8_) { // INT8
        data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
        printf("DATA_TYPE_EXT_1N_BYTE_SIGNED\n");
    } else { // FP32
        data_type = DATA_TYPE_EXT_FLOAT32;
        printf("DATA_TYPE_EXT_FLOAT32\n");
    }

    bm_ret = bm_image_create_batch (bm_handle_,
                                input_height,
                                input_width,
                                FORMAT_BGR_PLANAR,
                                data_type,
                                linear_trans_bmcv_,
                                max_batch_);

    if (BM_SUCCESS != bm_ret) {
        cout << "ERROR: bm_image_create_batch failed" << endl;
        exit(1);
    }
    // float input_scale = net_info_->input_scales[0]/255.0;
    float input_scale = net_info_->input_scales[0];
    printf("input_scale: %f\n",input_scale);

    linear_trans_param_.alpha_0 = input_scale;
    linear_trans_param_.beta_0 = 0;
    linear_trans_param_.alpha_1 = input_scale;
    linear_trans_param_.beta_1 = 0;
    linear_trans_param_.alpha_2 = input_scale;
    linear_trans_param_.beta_2 = 0;

}

int YOLOX::getInputBatchSize()
{
    return input_shape_.dims[0];
}


YOLOX::~YOLOX()
{
    delete output_;
    output_ = NULL;

    delete grids_x_;
    grids_x_ = NULL;
    delete grids_y_;
    grids_y_ = NULL;
    delete expanded_strides_;
    expanded_strides_ = NULL;

    // deinit bm images
    bm_image_destroy_batch (resize_bmcv_, max_batch_);
    bm_image_destroy_batch (linear_trans_bmcv_, max_batch_);
    
    delete resize_bmcv_;
    resize_bmcv_ = NULL;
    delete linear_trans_bmcv_;
    linear_trans_bmcv_ = NULL;

    bmrt_destroy(p_bmrt_);
}

int YOLOX::preForward(std::vector<bm_image> &input)
{
    LOG_TS(ts_, "YOLOX pre-process")
    if(max_batch_ < input.size()){
        printf("Error Input images more than batch size!\n");
        return -1;
    }
    preprocess_bmcv (input);
    LOG_TS(ts_, "YOLOX pre-process")
    return 0;
}

void YOLOX::forward(float threshold,float nms_threshold)
{
    threshold_ = threshold;
    nms_threshold_ = nms_threshold;
    LOG_TS(ts_, "YOLOX inference")
    bool res = bm_inference(p_bmrt_, linear_trans_bmcv_, (void*)output_, input_shape_, net_name_.c_str());

    LOG_TS(ts_, "YOLOX inference")

    if (!res) {
        cout << "ERROR : inference failed!!"<< endl;
        exit(1);
    }
}

void YOLOX::get_source_label(float* data_ptr, int classes, float &score, int &class_id)
{
	int i, max_i = 1;
    float* score_ptr = data_ptr+5;
	float max = score_ptr[0];
	for (i = 0; i < classes; ++i) {
		if (score_ptr[i] > max) {
			max = score_ptr[i];
			max_i = i;
		}
	}
    class_id = max_i;
    score = max*data_ptr[4];
}

float overlap_FM(float x1, float w1, float x2, float w2)
{
	float l1 = x1;
	float l2 = x2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1;
	float r2 = x2 + w2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float box_intersection_FM(ObjRect a, ObjRect b)
{
	float w = overlap_FM(a.left, a.width, b.left, b.width);
	float h = overlap_FM(a.top, a.height, b.top, b.height);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

float box_union_FM(ObjRect a, ObjRect b)
{
	float i = box_intersection_FM(a, b);
	float u = a.width*a.height + b.width*b.height - i;
	return u;
}

float box_iou_FM(ObjRect a, ObjRect b)
{
	return box_intersection_FM(a, b) / box_union_FM(a, b);
}

static bool sort_ObjRect(ObjRect a, ObjRect b)
{
    return a.score > b.score;
}

static void nms_sorted_bboxes(const std::vector<ObjRect>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = objects.size();

    for (int i = 0; i < n; i++)    {
        const ObjRect& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const ObjRect& b = objects[picked[j]];

            float iou = box_iou_FM(a, b);
            if (iou > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

int save_featuremap(float* m_data,int outputs, int layer_w,int layer_h, char* save_name)
{
	FILE* fp_temp = fopen(save_name, "w+");
	int line_num = 0;
	for (int mm = 0; mm < outputs;mm++)
	{
		char txt_temp[32] = { 0 };
		if (layer_w >= 1){
			if (mm % layer_w == layer_w - 1)
			{
				sprintf(txt_temp, "%.2f\n", m_data[mm]);
				line_num++;
				if (line_num % layer_h == 0){
					sprintf(txt_temp, "%.2f\n", m_data[mm]);
				}
			}
			else
			{
				sprintf(txt_temp, "%.2f ", m_data[mm]);
			}
		}
		else{
			sprintf(txt_temp, "%.2f ", m_data[mm]);
		}
		fputs(txt_temp, fp_temp);
	}
	fclose(fp_temp);
	return 0;
}

int save_featuremap(char* m_data,int outputs, int layer_w,int layer_h, char* save_name, float scale)
{
	FILE* fp_temp = fopen(save_name, "w+");
	int line_num = 0;
	for (int mm = 0; mm < outputs;mm++)
	{
		char txt_temp[32] = { 0 };
		if (layer_w >= 1){
			if (mm % layer_w == layer_w - 1)
			{
				sprintf(txt_temp, "%.2f\n", scale*m_data[mm]);
				line_num++;
				if (line_num % layer_h == 0){
					sprintf(txt_temp, "%.2f\n", scale*m_data[mm]);
				}
			}
			else
			{
				sprintf(txt_temp, "%.2f ", scale*m_data[mm]);
			}
		}
		else{
			sprintf(txt_temp, "%.2f ", scale*m_data[mm]);
		}
		fputs(txt_temp, fp_temp);
	}
	fclose(fp_temp);
	return 0;
}

int save_featuremap(char* m_data,int outputs, int layer_w,int layer_h, char* save_name)
{
	FILE* fp_temp = fopen(save_name, "w+");
	int line_num = 0;
	for (int mm = 0; mm < outputs;mm++)
	{
		char txt_temp[32] = { 0 };
		if (layer_w >= 1){
			if (mm % layer_w == layer_w - 1)
			{
				sprintf(txt_temp, "%d\n", m_data[mm]);
				line_num++;
				if (line_num % layer_h == 0){
					sprintf(txt_temp, "%d\n", m_data[mm]);
				}
			}
			else
			{
				sprintf(txt_temp, "%d ", m_data[mm]);
			}
		}
		else{
			sprintf(txt_temp, "%d ", m_data[mm]);
		}
		fputs(txt_temp, fp_temp);
	}
	fclose(fp_temp);
	return 0;
}


void YOLOX::postForward(std::vector<bm_image> &input, std::vector<std::vector<ObjRect>> &detections)
{
    detections.clear();
    int size_one_batch = output_size / max_batch_;
    if(output_is_int8_){
        // save_featuremap((char*)output_,size_one_batch, 85,1, "int8.txt");
        // save_featuremap((char*)output_,size_one_batch, 85,1, "int8.txt",output_scale_);
        char* m_data_ptr = (char*)output_;
         for (int batch_idx=0; batch_idx<input.size();batch_idx++){
            float* output_start = output_ + (size_one_batch * batch_idx)*4;
            int batch_start_ptr = size_one_batch * batch_idx;
            std::vector<ObjRect> dect_temp;
            dect_temp.clear();
            float scale_x = (float)input[batch_idx].width/input_width;
            float scale_y = (float)input[batch_idx].height/input_height;
            for (size_t i = 0; i < outlen_dim1_; i++)    {
                int ptr_start=i*channels_resu_;
                float box_objectness = output_scale_ * m_data_ptr[batch_start_ptr + ptr_start+4];
                if(box_objectness >= threshold_){
                    float center_x = (output_scale_ * m_data_ptr[batch_start_ptr +ptr_start+0] + grids_x_[i]) * expanded_strides_[i];
                    float center_y = (output_scale_ * m_data_ptr[batch_start_ptr +ptr_start+1] + grids_y_[i]) * expanded_strides_[i];
                    float w_temp = exp(output_scale_ * m_data_ptr[batch_start_ptr +ptr_start+2])*expanded_strides_[i];
                    float h_temp = exp(output_scale_ * m_data_ptr[batch_start_ptr +ptr_start+3])*expanded_strides_[i];
                    float score = output_scale_ * m_data_ptr[batch_start_ptr +ptr_start+4];
                    center_x *= scale_x;
                    center_y *= scale_y;
                    w_temp *= scale_x;
                    h_temp *= scale_y;
                    float left = center_x - w_temp/2;
                    float top = center_y - h_temp/2;
                    float right = center_x + w_temp/2;
                    float bottom = center_y + h_temp/2;

                    // printf("[%.2f,%.2f,%.2f,%.2f]::::%f,%f\n",center_x,center_y,w_temp,h_temp,scale_x,scale_y);

                    for (int class_idx = 0; class_idx < classes_; class_idx++)       {
                        float box_cls_score = output_scale_ * m_data_ptr[batch_start_ptr +ptr_start + 5 + class_idx];
                        float box_prob = box_objectness * box_cls_score;
                        if (box_prob > threshold_)         {
                            ObjRect obj_temp;
                            obj_temp.width = w_temp;
                            obj_temp.height = h_temp;
                            obj_temp.left = left;
                            obj_temp.top = top;
                            obj_temp.right = right;
                            obj_temp.bottom = bottom;
                            obj_temp.score = box_prob;
                            obj_temp.class_id = class_idx;
                            dect_temp.push_back(obj_temp);
                        }
                    }
                }
            }

            std::sort(dect_temp.begin(),dect_temp.end(),sort_ObjRect);

            std::vector<ObjRect> dect_temp_batch;
            std::vector<int> picked;
            dect_temp_batch.clear();
            nms_sorted_bboxes(dect_temp, picked, nms_threshold_);

            // for (size_t idx_xx = 0; idx_xx < dect_temp.size(); idx_xx++)
            // {
            //     printf("%d:%.0f,%.0f,%.0f,%.0f\n",batch_idx,dect_temp[idx_xx].left,dect_temp[idx_xx].top,dect_temp[idx_xx].right,dect_temp[idx_xx].bottom);
            // }
            // printf("##############################################################\n");

            for (size_t i = 0; i < picked.size(); i++)    {
                dect_temp_batch.push_back(dect_temp[picked[i]]);
            }
            
            detections.push_back(dect_temp_batch);
        }

    }else{
        // save_featuremap(output_,size_one_batch, 85,1, "fp32.txt");
        for (int batch_idx=0; batch_idx<input.size();batch_idx++){
            float* output_start = output_ + (size_one_batch * batch_idx)*4;
            int batch_start_ptr = size_one_batch * batch_idx;
            std::vector<ObjRect> dect_temp;
            dect_temp.clear();
            float scale_x = (float)input[batch_idx].width/input_width;
            float scale_y = (float)input[batch_idx].height/input_height;
            for (size_t i = 0; i < outlen_dim1_; i++)    {
                int ptr_start=i*channels_resu_;
                float box_objectness = output_[batch_start_ptr + ptr_start+4];
                if(output_[batch_start_ptr + ptr_start+4] >= threshold_){
                    float center_x = (output_[batch_start_ptr +ptr_start+0] + grids_x_[i]) * expanded_strides_[i];
                    float center_y = (output_[batch_start_ptr +ptr_start+1] + grids_y_[i]) * expanded_strides_[i];
                    float w_temp = exp(output_[batch_start_ptr +ptr_start+2])*expanded_strides_[i];
                    float h_temp = exp(output_[batch_start_ptr +ptr_start+3])*expanded_strides_[i];
                    float score = output_[batch_start_ptr +ptr_start+4];
                    center_x *= scale_x;
                    center_y *= scale_y;
                    w_temp *= scale_x;
                    h_temp *= scale_y;
                    float left = center_x - w_temp/2;
                    float top = center_y - h_temp/2;
                    float right = center_x + w_temp/2;
                    float bottom = center_y + h_temp/2;

                    // printf("[%.2f,%.2f,%.2f,%.2f]::::%f,%f\n",center_x,center_y,w_temp,h_temp,scale_x,scale_y);

                    for (int class_idx = 0; class_idx < classes_; class_idx++)       {
                        float box_cls_score = output_[batch_start_ptr +ptr_start + 5 + class_idx];
                        float box_prob = box_objectness * box_cls_score;
                        if (box_prob > threshold_)         {
                            ObjRect obj_temp;
                            obj_temp.width = w_temp;
                            obj_temp.height = h_temp;
                            obj_temp.left = left;
                            obj_temp.top = top;
                            obj_temp.right = right;
                            obj_temp.bottom = bottom;
                            obj_temp.score = box_prob;
                            obj_temp.class_id = class_idx;
                            dect_temp.push_back(obj_temp);
                        }
                    }
                }
            }

            std::sort(dect_temp.begin(),dect_temp.end(),sort_ObjRect);

            std::vector<ObjRect> dect_temp_batch;
            std::vector<int> picked;
            dect_temp_batch.clear();
            nms_sorted_bboxes(dect_temp, picked, nms_threshold_);

            // for (size_t idx_xx = 0; idx_xx < dect_temp.size(); idx_xx++)
            // {
            //     printf("%d:%.0f,%.0f,%.0f,%.0f\n",batch_idx,dect_temp[idx_xx].left,dect_temp[idx_xx].top,dect_temp[idx_xx].right,dect_temp[idx_xx].bottom);
            // }
            // printf("##############################################################\n");

            for (size_t i = 0; i < picked.size(); i++)    {
                dect_temp_batch.push_back(dect_temp[picked[i]]);
            }
            
            detections.push_back(dect_temp_batch);
        }
    }
}

void YOLOX::enableProfile(TimeStamp *ts)
{
    ts_ = ts;
}

bool YOLOX::getPrecision()
{
  return is_int8_;
}

void YOLOX::preprocess_bmcv(std::vector<bm_image> &input)
{
    if(input.empty()){
        printf("mul-batch bmcv input empty!!!\n");
        return ;
    }
    if(input.size() != 1 && input.size() != 4){
        printf("mul-batch input error!\n");
        return ;
    }


    // resize && split by bmcv
    for (size_t i = 0; i < input.size(); i++) {
        LOG_TS(ts_, "YOLOX pre-process-vpp")
        crop_rect_ = {0, 0, input[i].width, input[i].height};
        bmcv_image_vpp_convert (bm_handle_, 1, input[i], &resize_bmcv_[i], &crop_rect_);
        LOG_TS(ts_, "YOLOX pre-process-vpp")
    }

    // do linear transform
    LOG_TS(ts_, "YOLOX pre-process-linear_tranform")
    bmcv_image_convert_to (bm_handle_, input.size(), linear_trans_param_, resize_bmcv_, linear_trans_bmcv_);
    LOG_TS(ts_, "YOLOX pre-process-linear_tranform")
}
