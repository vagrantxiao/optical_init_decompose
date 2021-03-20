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
#include "tensor_weight_y.h"
#include "tensor_weight_x.h"


// define these constants so they can be used in pragma
const int max_width = MAX_WIDTH; 
const int default_depth = MAX_WIDTH;







// compute output flow
void flow_calc(hls::stream< bit32> & Input_1,
               velocity_t outputs[MAX_HEIGHT][MAX_WIDTH])
{
  static outer_pixel_t buf[2];
  bit32 in_tmp;

  FLOW_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    FLOW_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      tensor_t tmp_tensor;
      in_tmp = Input_1.read();
      tmp_tensor.val[0](31,  0) = in_tmp(31,  0);
      in_tmp = Input_1.read();
      tmp_tensor.val[0](47, 32) = in_tmp(15,  0);
      tmp_tensor.val[1](15,  0) = in_tmp(31, 16);
      in_tmp = Input_1.read();
      tmp_tensor.val[1](47, 16) = in_tmp(31,  0);

      in_tmp = Input_1.read();
      tmp_tensor.val[2](31,  0) = in_tmp(31,  0);
      in_tmp = Input_1.read();
      tmp_tensor.val[2](47, 32) = in_tmp(15,  0);
      tmp_tensor.val[3](15,  0) = in_tmp(31, 16);
      in_tmp = Input_1.read();
      tmp_tensor.val[3](47, 16) = in_tmp(31,  0);


      in_tmp = Input_1.read();
      tmp_tensor.val[4](31,  0) = in_tmp(31,  0);
      in_tmp = Input_1.read();
      tmp_tensor.val[4](47, 32) = in_tmp(15,  0);
      tmp_tensor.val[5](15,  0) = in_tmp(31, 16);
      in_tmp = Input_1.read();
      tmp_tensor.val[5](47, 16) = in_tmp(31,  0);


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
  hls::stream< bit32 > tensor;

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
