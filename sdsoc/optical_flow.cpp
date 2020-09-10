/*===============================================================*/
/*                                                               */
/*                      optical_flow.cpp                         */
/*                                                               */
/*             Hardware function for optical flow                */
/*                                                               */
/*===============================================================*/

#include "optical_flow.h"

// use HLS video library
#include <hls_video.h>

// define these constants so they can be used in pragma
const int max_width = MAX_WIDTH; 
const int default_depth = 1;

// calculate gradient in x and y directions
void gradient_xy_calc(hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Output_1,
    pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH])
{
  #pragma HLS interface ap_fifo port=frame
  #pragma HLS interface ap_fifo port=gradient_x
  #pragma HLS interface ap_fifo port=gradient_y

  // our own line buffer
  static pixel_t buf[5][MAX_WIDTH];
  #pragma HLS array_partition variable=buf complete dim=1

  // small buffer
  pixel_t smallbuf[5];
  #pragma HLS array_partition variable=smallbuf complete dim=0
  
  // window buffer
  hls::Window<5,5,pixel_t> window;

  const int GRAD_WEIGHTS[] =  {1,-8,0,8,-1};

  GRAD_XY_OUTER: for(int r=0; r<MAX_HEIGHT+2; r++)
  {
    GRAD_XY_INNER: for(int c=0; c<MAX_WIDTH+2; c++)
    {
      #pragma HLS pipeline II=1

      // read out values from current line buffer
      for (int i = 0; i < 4; i ++ )
        smallbuf[i] = buf[i+1][c];
      // the new value is either 0 or read from frame
      if (r<MAX_HEIGHT && c<MAX_WIDTH)
      {
    	bit32 buf = Input_1.read();
        smallbuf[4] = (pixel_t)(buf(23, 16));
      }
      else if (c < MAX_WIDTH)
        smallbuf[4] = 0.0f;
      // update line buffer
      if(r<MAX_HEIGHT && c<MAX_WIDTH)
      {
        for (int i = 0; i < 4; i ++ )
          buf[i][c] = smallbuf[i];
        buf[4][c] = smallbuf[4];
      }
      else if(c<MAX_WIDTH)
      {
        for (int i = 0; i < 4; i ++ )
          buf[i][c] = smallbuf[i];
        buf[4][c] = smallbuf[4];
      }

      // manage window buffer
      if(r<MAX_HEIGHT && c<MAX_WIDTH)
      {
        window.shift_pixels_left();
        
        for (int i = 0; i < 5; i ++ )
          window.insert_pixel(smallbuf[i],i,4);
      }
      else
      {
        window.shift_pixels_left();
        window.insert_pixel(0,0,4);
        window.insert_pixel(0,1,4);
        window.insert_pixel(0,2,4);
        window.insert_pixel(0,3,4);
        window.insert_pixel(0,4,4);
      }

      // compute gradient
      pixel_t x_grad = 0;
      pixel_t y_grad = 0;
      if(r>=4 && r<MAX_HEIGHT && c>=4 && c<MAX_WIDTH)
      {
        GRAD_XY_XYGRAD: for(int i=0; i<5; i++)
        {
          x_grad += window.getval(2,i)*GRAD_WEIGHTS[i];
          y_grad += window.getval(i,2)*GRAD_WEIGHTS[i];
        }
        //gradient_x[r-2][c-2] = x_grad/12;
        Output_1.write(x_grad/12);
        gradient_y[r-2][c-2] = y_grad/12;
      }
      else if(r>=2 && c>=2)
      {
        //gradient_x[r-2][c-2] = 0;
    	Output_1.write(0);
        gradient_y[r-2][c-2] = 0;
      }
    }
  }
}

// calculate gradient in the z direction
void gradient_z_calc(
		hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Input_2,
    pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH])
{
  #pragma HLS interface ap_fifo port=frame1
  #pragma HLS interface ap_fifo port=frame2
  #pragma HLS interface ap_fifo port=frame3
  #pragma HLS interface ap_fifo port=frame4
  #pragma HLS interface ap_fifo port=frame5
  #pragma HLS interface ap_fifo port=gradient_z

  const int GRAD_WEIGHTS[] =  {1,-8,0,8,-1};
  GRAD_Z_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    GRAD_Z_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      frames_t buf;
      buf = Input_1.read();
      pixel_t in1 = (pixel_t)(buf(7 ,  0));
      pixel_t in2 = (pixel_t)(buf(15,  8));
      pixel_t in3 = (pixel_t)(buf(23, 16));
      pixel_t in4 = (pixel_t)(buf(31, 24));
      buf = Input_2.read();
      pixel_t in5 = (pixel_t)(buf(7 ,  0));
      gradient_z[r][c] = (in1*GRAD_WEIGHTS[0]
                        + in2*GRAD_WEIGHTS[1]
                        + in3*GRAD_WEIGHTS[2]
                        + in4*GRAD_WEIGHTS[3]
                        + in5*GRAD_WEIGHTS[4])/12;
    }
  }
}

// average the gradient in y direction
void gradient_weight_y(hls::stream<bit32> & Input_1,
    pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH],
    gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH])
{
  #pragma HLS interface ap_fifo port=gradient_x
  #pragma HLS interface ap_fifo port=gradient_y
  #pragma HLS interface ap_fifo port=gradient_z
  #pragma HLS interface ap_fifo port=filt_grad
  hls::LineBuffer<7,MAX_WIDTH,gradient_t> buf;

  const pixel_t GRAD_FILTER[] = {0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755};
  GRAD_WEIGHT_Y_OUTER: for(int r=0; r<MAX_HEIGHT+3; r++)
  {
    GRAD_WEIGHT_Y_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
            
      if(r<MAX_HEIGHT)
      {
        buf.shift_pixels_up(c);
        gradient_t tmp;
        tmp.x = (pixel_t)Input_1.read();
        tmp.y = gradient_y[r][c];
        tmp.z = gradient_z[r][c];
        buf.insert_bottom_row(tmp,c);
      }
      else
      {
        buf.shift_pixels_up(c);
        gradient_t tmp;
        tmp.x = 0;
        tmp.y = 0;
        tmp.z = 0;
        buf.insert_bottom_row(tmp,c);
      }     

      gradient_t acc;
      acc.x = 0;
      acc.y = 0;
      acc.z = 0;
      if(r >= 6 && r<MAX_HEIGHT)
      { 
        GRAD_WEIGHT_Y_ACC: for(int i=0; i<7; i++)
        {
          acc.x += buf.getval(i,c).x*GRAD_FILTER[i];
          acc.y += buf.getval(i,c).y*GRAD_FILTER[i];
          acc.z += buf.getval(i,c).z*GRAD_FILTER[i];
        }
        filt_grad[r-3][c] = acc;            
      }
      else if(r>=2)
      {
        filt_grad[r-3][c] = acc;
      }
    }
  }
}

// average gradient in the x direction
void gradient_weight_x(gradient_t y_filt[MAX_HEIGHT][MAX_WIDTH],
                       gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH])
{
  #pragma HLS interface ap_fifo port=y_filt
  #pragma HLS interface ap_fifo port=filt_grad
  hls::Window<1,7,gradient_t> buf;
  const pixel_t GRAD_FILTER[] = {0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755};
  GRAD_WEIGHT_X_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    GRAD_WEIGHT_X_INNER: for(int c=0; c<MAX_WIDTH+3; c++)
    {
      #pragma HLS pipeline II=1
      buf.shift_pixels_left();
      gradient_t tmp;
      if(c<MAX_WIDTH)
      {
        tmp = y_filt[r][c];
      }
      else
      {
        tmp.x = 0;
        tmp.y = 0;
        tmp.z = 0;  
      }
      buf.insert_pixel(tmp,0,6);

      gradient_t acc;
      acc.x = 0;
      acc.y = 0;
      acc.z = 0;
      if(c >= 6 && c<MAX_WIDTH) 
      {
        GRAD_WEIGHT_X_ACC: for(int i=0; i<7; i++)
        {
          acc.x += buf.getval(0,i).x*GRAD_FILTER[i];
          acc.y += buf.getval(0,i).y*GRAD_FILTER[i];
          acc.z += buf.getval(0,i).z*GRAD_FILTER[i];
        }
        filt_grad[r][c-3] = acc;
      }
      else if(c>=3)
      {
        filt_grad[r][c-3] = acc;
      }
    }
  }
}

// outer product 
void outer_product(gradient_t gradient[MAX_HEIGHT][MAX_WIDTH],
     outer_t outer_product[MAX_HEIGHT][MAX_WIDTH])
{
  #pragma HLS interface ap_fifo port=gradient
  #pragma HLS interface ap_fifo port=outer_product
    
  OUTER_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    OUTER_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      gradient_t grad = gradient[r][c];
      outer_t out;
      out.val[0] = grad.x*grad.x;
      out.val[1] = grad.y*grad.y;
      out.val[2] = grad.z*grad.z;
      out.val[3] = grad.x*grad.y;
      out.val[4] = grad.x*grad.z;
      out.val[5] = grad.y*grad.z;
      outer_product[r][c] = out;
    }
  }
}

// tensor weight
void tensor_weight_y(outer_t outer[MAX_HEIGHT][MAX_WIDTH],
                     tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH])
{
  #pragma HLS interface ap_fifo port=outer
  #pragma HLS interface ap_fifo port=tensor_y
  hls::LineBuffer<3,MAX_WIDTH,outer_t> buf;
  const pixel_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
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
        tmp = outer[r][c];
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
        tensor_y[r-1][c] = acc;      
      }
    }
  }
}

void tensor_weight_x(tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH],
                     tensor_t tensor[MAX_HEIGHT][MAX_WIDTH])
{
  #pragma HLS interface ap_fifo port=tensor_y
  #pragma HLS interface ap_fifo port=tensor
  hls::Window<1,3,tensor_t> buf;
  const pixel_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  TENSOR_WEIGHT_X_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    TENSOR_WEIGHT_X_INNER: for(int c=0; c<MAX_WIDTH+1; c++)
    {
      #pragma HLS pipeline II=1
      buf.shift_pixels_left();
      tensor_t tmp;
      if(c<MAX_WIDTH)
      {
        tmp = tensor_y[r][c];
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
  #pragma HLS interface ap_fifo port=tensors

  static pixel_t buf[2];

  FLOW_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    FLOW_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      if(r>=2 && r<MAX_HEIGHT-2 && c>=2 && c<MAX_WIDTH-2)
      {
        pixel_t denom = tensors[r][c].val[0]*tensors[r][c].val[1]-
                        tensors[r][c].val[3]*tensors[r][c].val[3];
        buf[0] = (tensors[r][c].val[5]*tensors[r][c].val[3]-
                 tensors[r][c].val[4]*tensors[r][c].val[1]) / denom;
        buf[1] = (tensors[r][c].val[4]*tensors[r][c].val[3]-
                 tensors[r][c].val[5]*tensors[r][c].val[0]) / denom;
      }
      else
      {
        buf[0] = buf[1] = 0;
      }

      outputs[r][c].x = buf[0];
      outputs[r][c].y = buf[1];

    }
  }
}

// top-level kernel function
void optical_flow(hls::stream<frames_t> & Input_1,
                  velocity_t outputs[MAX_HEIGHT][MAX_WIDTH])
{
  #pragma HLS data_pack variable=outputs

  #pragma HLS DATAFLOW
	hls::stream<bit32> unpack_out_1;
	hls::stream<bit32> unpack_out_2;
	hls::stream<bit32> unpack_out_3;
	hls::stream<bit32> unpack_out_4;
	hls::stream<bit32> unpack_out_5;
	hls::stream<bit32> unpack_out_6;
	hls::stream<bit32> gradient_xy_calc_out1;

  // FIFOs connecting the stages
  static pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=gradient_x depth=default_depth
  static pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=gradient_y depth=default_depth
  static pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=gradient_z depth=max_width*4
  static gradient_t y_filtered[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=y_filtered depth=default_depth
  static gradient_t filtered_gradient[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=filtered_gradient depth=default_depth
  static outer_t out_product[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=out_product depth=default_depth
  #pragma HLS data_pack variable=out_product
  static tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=tensor_y depth=default_depth
  #pragma HLS data_pack variable=tensor_y
  static tensor_t tensor[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=tensor depth=default_depth
  #pragma HLS data_pack variable=tensor

  // FIFOs for streaming in, just for clarity
  static pixel_t frame1_a[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame1_a depth=default_depth
  static pixel_t frame2_a[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame2_a depth=default_depth
  static pixel_t frame4_a[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame4_a depth=default_depth
  static pixel_t frame5_a[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame5_a depth=default_depth

  // Need to duplicate frame3 for the two calculations
  static pixel_t frame3_a[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame3_a depth=default_depth
  static pixel_t frame3_b[MAX_HEIGHT][MAX_WIDTH];
  #pragma HLS STREAM variable=frame3_b depth=default_depth

  // stream in and organize the inputs
  static frames_t buf;
  FRAMES_CP_OUTER: for (int r=0; r<MAX_HEIGHT; r++) 
  {
    FRAMES_CP_INNER: for (int c=0; c<MAX_WIDTH; c++) 
    {
      #pragma HLS pipeline II=1

      // one wide read
      buf = Input_1.read();
      // assign values to the FIFOs
      //frame3_a[r][c] = (pixel_t)(buf(23, 16)) / 255.0f;
      unpack_out_1.write(buf(31, 0));

      //frame1_a[r][c] = (pixel_t)(buf(7 ,  0)) / 255.0f;
      unpack_out_2.write(buf(31, 0));

      //frame2_a[r][c] = (pixel_t)(buf(15,  8)) / 255.0f;

      //frame3_b[r][c] = (pixel_t)(buf(23, 16)) / 255.0f;

      //frame4_a[r][c] = (pixel_t)(buf(31, 24)) / 255.0f;

      //frame5_a[r][c] = (pixel_t)(buf(39, 32)) / 255.0f;
      unpack_out_3.write(buf(63, 32));
    }
  }

  // compute 
  gradient_xy_calc(unpack_out_1, gradient_xy_calc_out1, gradient_y);
  gradient_z_calc(unpack_out_2, unpack_out_3, gradient_z);
  gradient_weight_y(gradient_xy_calc_out1, gradient_y, gradient_z, y_filtered);
  gradient_weight_x(y_filtered, filtered_gradient);
  outer_product(filtered_gradient, out_product);
  tensor_weight_y(out_product, tensor_y);
  tensor_weight_x(tensor_y, tensor);
  flow_calc(tensor, outputs);

}
