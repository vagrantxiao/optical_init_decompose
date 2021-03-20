/*===============================================================*/
/*                                                               */
/*                      optical_flow.cpp                         */
/*                                                               */
/*             Hardware function for optical flow                */
/*                                                               */
/*===============================================================*/

#include "optical_flow.h"
// use HLS video library


// use HLS fixed point
#include "ap_fixed.h"
#include "unpack.h"
#include "gradient_xy_calc.h"
#include "gradient_z_calc.h"
#include "gradient_weight_y.h"
#include "gradient_weight_x.h"
#include "outer_product.h"


// define these constants so they can be used in pragma
const int max_width = MAX_WIDTH; 
const int default_depth = MAX_WIDTH;





// tensor weight
void tensor_weight_y(hls::stream< bit32> & Input_1,
		hls::stream< bit32> & Output_1)
{
  hls::LineBuffer<3,MAX_WIDTH,outer_t> buf;
  const pixel_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  bit32 in_tmp;
  bit32 out_tmp;

  TENSOR_WEIGHT_Y_OUTER: for(int r=0; r<MAX_HEIGHT+1; r++)
  {
    TENSOR_WEIGHT_Y_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      
      outer_t tmp;
      #pragma HLS data_pack variable=tmp
      #pragma HLS data_pack variable=buf.val[0]
      buf.shift_pixels_up(c);
      if(r<MAX_HEIGHT)
      {
        in_tmp = Input_1.read();
        tmp.val[0](31,  0) = in_tmp(31,  0);
        in_tmp = Input_1.read();
        tmp.val[0](47, 32) = in_tmp(15,  0);
        tmp.val[1](15,  0) = in_tmp(31, 16);
        in_tmp = Input_1.read();
        tmp.val[1](47, 16) = in_tmp(31,  0);

        in_tmp = Input_1.read();
        tmp.val[2](31,  0) = in_tmp(31,  0);
        in_tmp = Input_1.read();
        tmp.val[2](47, 32) = in_tmp(15,  0);
        tmp.val[3](15,  0) = in_tmp(31, 16);
        in_tmp = Input_1.read();
        tmp.val[3](47, 16) = in_tmp(31,  0);


        in_tmp = Input_1.read();
        tmp.val[4](31,  0) = in_tmp(31,  0);
        in_tmp = Input_1.read();
        tmp.val[4](47, 32) = in_tmp(15,  0);
        tmp.val[5](15,  0) = in_tmp(31, 16);
        in_tmp = Input_1.read();
        tmp.val[5](47, 16) = in_tmp(31,  0);


      }
      else
      {
        TENSOR_WEIGHT_Y_TMP_INIT: for(int i=0; i<6; i++)
          tmp.val[i] = 0;
      }   
      buf.insert_bottom_row(tmp,c);

      tensor_t acc;
      TENSOR_WEIGHT_Y_ACC_INIT: for(int k =0; k<6; k++)
        acc.val[k] = 0;
     
      if (r >= 2 && r < MAX_HEIGHT) 
      {
        TENSOR_WEIGHT_Y_TMP_OUTER: for(int i=0; i<3; i++)
        {
          tmp = buf.getval(i,c);
          pixel_t k = TENSOR_FILTER[i];
          TENSOR_WEIGHT_Y_TMP_INNER: for(int component=0; component<6; component++)
          {
            acc.val[component] += tmp.val[component]*k;
          }
        }
      }
      if(r >= 1)
      { 
        //tensor_y[r-1][c] = acc;
        out_tmp(31,  0) = acc.val[0](31,  0);
        Output_1.write(out_tmp);
        out_tmp(15,  0) = acc.val[0](47, 32);
        out_tmp(31, 16) = acc.val[1](15,  0);
        Output_1.write(out_tmp);
        out_tmp(31,  0) = acc.val[1](47, 16);
        Output_1.write(out_tmp);

        out_tmp(31,  0) = acc.val[2](31,  0);
        Output_1.write(out_tmp);
        out_tmp(15,  0) = acc.val[2](47, 32);
        out_tmp(31, 16) = acc.val[3](15,  0);
        Output_1.write(out_tmp);
        out_tmp(31,  0) = acc.val[3](47, 16);
        Output_1.write(out_tmp);

        out_tmp(31,  0) = acc.val[4](31,  0);
        Output_1.write(out_tmp);
        out_tmp(15,  0) = acc.val[4](47, 32);
        out_tmp(31, 16) = acc.val[5](15,  0);
        Output_1.write(out_tmp);
        out_tmp(31,  0) = acc.val[5](47, 16);
        Output_1.write(out_tmp);

      }
    }
  }
}

void tensor_weight_x(hls::stream< bit32> & Input_1,
                     tensor_t tensor[MAX_HEIGHT][MAX_WIDTH])
{
  bit32 in_tmp;
  hls::Window<1,3,tensor_t> buf;
  const pixel_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  //const float TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  TENSOR_WEIGHT_X_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    TENSOR_WEIGHT_X_INNER: for(int c=0; c<MAX_WIDTH+1; c++)
    {
      #pragma HLS pipeline II=1
      buf.shift_pixels_left();
      tensor_t tmp;
      if(c<MAX_WIDTH)
      {
        //tmp = tensor_y[r][c];
          in_tmp = Input_1.read();
          tmp.val[0](31,  0) = in_tmp(31,  0);
          in_tmp = Input_1.read();
          tmp.val[0](47, 32) = in_tmp(15,  0);
          tmp.val[1](15,  0) = in_tmp(31, 16);
          in_tmp = Input_1.read();
          tmp.val[1](47, 16) = in_tmp(31,  0);

          in_tmp = Input_1.read();
          tmp.val[2](31,  0) = in_tmp(31,  0);
          in_tmp = Input_1.read();
          tmp.val[2](47, 32) = in_tmp(15,  0);
          tmp.val[3](15,  0) = in_tmp(31, 16);
          in_tmp = Input_1.read();
          tmp.val[3](47, 16) = in_tmp(31,  0);


          in_tmp = Input_1.read();
          tmp.val[4](31,  0) = in_tmp(31,  0);
          in_tmp = Input_1.read();
          tmp.val[4](47, 32) = in_tmp(15,  0);
          tmp.val[5](15,  0) = in_tmp(31, 16);
          in_tmp = Input_1.read();
          tmp.val[5](47, 16) = in_tmp(31,  0);
      }
      else
      {
        TENSOR_WEIGHT_X_TMP_INIT: for(int i=0; i<6; i++)
          tmp.val[i] = 0;
      }
      buf.insert_pixel(tmp,0,2);

      tensor_t acc;
      TENSOR_WEIGHT_X_ACC_INIT: for(int k =0; k<6; k++)
        acc.val[k] = 0;
      if (c >= 2 && c < MAX_WIDTH) 
      {
        TENSOR_WEIGHT_X_TMP_OUTER: for(int i=0; i<3; i++)
        {
          tmp = buf.getval(0,i);
          TENSOR_WEIGHT_X_TMP_INNER: for(int component=0; component<6; component++)
          {
            acc.val[component] += tmp.val[component]*TENSOR_FILTER[i];
          }
        }
      }
      if(c>=1)
      {
        tensor[r][c-1] = acc;
      }
    }
  }
}

// compute output flow
void flow_calc(tensor_t tensors[MAX_HEIGHT][MAX_WIDTH],
               velocity_t outputs[MAX_HEIGHT][MAX_WIDTH])
{
  static outer_pixel_t buf[2];
  FLOW_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    FLOW_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      tensor_t tmp_tensor = tensors[r][c];
      if(r>=2 && r<MAX_HEIGHT-2 && c>=2 && c<MAX_WIDTH-2)
      {
	      calc_pixel_t t1 = (calc_pixel_t) tmp_tensor.val[0];
	      calc_pixel_t t2 = (calc_pixel_t) tmp_tensor.val[1];
	      calc_pixel_t t3 = (calc_pixel_t) tmp_tensor.val[2];
	      calc_pixel_t t4 = (calc_pixel_t) tmp_tensor.val[3];
	      calc_pixel_t t5 = (calc_pixel_t) tmp_tensor.val[4];
	      calc_pixel_t t6 = (calc_pixel_t) tmp_tensor.val[5];

        calc_pixel_t denom = t1*t2-t4*t4;
	      calc_pixel_t numer0 = t6*t4-t5*t2;
	      calc_pixel_t numer1 = t5*t4-t6*t1;

	      if(denom != 0)
        {
          buf[0] = numer0 / denom;
          buf[1] = numer1 / denom;
	      }
	      else
	      {
		      buf[0] = 0;
		      buf[1] = 0;
	      }
      }
      else
      {
        buf[0] = buf[1] = 0;
      }

      outputs[r][c].x = (vel_pixel_t)buf[0];
      outputs[r][c].y = (vel_pixel_t)buf[1];

    }
  }
}

// top-level kernel function
void optical_flow(hls::stream<frames_t> & Input_1,
                  velocity_t outputs[MAX_HEIGHT][MAX_WIDTH])
{
  #pragma HLS data_pack variable=outputs

  #pragma HLS DATAFLOW

  // FIFOs connecting the stages
  #pragma HLS data_pack variable=tensor_y
  static tensor_t tensor[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=tensor depth=default_depth
  #pragma HLS data_pack variable=tensor

  // FIFOs for streaming in, just for clarity

  //Need to duplicate frame3 for the two calculations
  hls::stream< bit32 > frame3_a;
  hls::stream< bit32 > frame1_a;
  hls::stream< bit32 > frame2_a;
  hls::stream< bit32 > frame4_a;
  hls::stream< bit32 > frame5_a;
  hls::stream< bit32 > frame3_b;
  hls::stream< bit32 > gradient_x;
  hls::stream< bit32 > gradient_y;
  hls::stream< bit32 > gradient_z;
  hls::stream< bit32 > y_filtered;
  hls::stream< bit32 > filtered_gradient;
  hls::stream< bit32 > out_product;
  hls::stream< bit32 > tensor_y;

  unpack(Input_1, frame1_a, frame2_a, frame4_a, frame5_a, frame3_a, frame3_b);
  //
  // compute
  gradient_xy_calc(frame3_a, gradient_x, gradient_y);
  gradient_z_calc(frame1_a, frame2_a, frame3_b, frame4_a, frame5_a, gradient_z);
  gradient_weight_y(gradient_x, gradient_y, gradient_z, y_filtered);
  gradient_weight_x(y_filtered, filtered_gradient);
  outer_product(filtered_gradient, out_product);
  tensor_weight_y(out_product, tensor_y);
  tensor_weight_x(tensor_y, tensor);
  flow_calc(tensor, outputs);

}
