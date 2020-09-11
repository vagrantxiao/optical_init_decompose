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

// use HLS fixed point
#include "ap_fixed.h"

// define these constants so they can be used in pragma
const int max_width = MAX_WIDTH; 
const int default_depth = MAX_WIDTH;
// calculate gradient in x and y directions
void gradient_xy_calc(
		hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Output_1,
		hls::stream<bit32> & Output_2
		)
{
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_2
#pragma HLS INTERFACE ap_hs port=Output_1

  // our own line buffer
  static pixel_t buf[5][MAX_WIDTH];
  #pragma HLS array_partition variable=buf complete dim=1

  // small buffer
  pixel_t smallbuf[5];
  #pragma HLS array_partition variable=smallbuf complete dim=0
  
  // window buffer
  hls::Window<5,5,input_t> window;
  const int GRAD_WEIGHTS[] =  {1,-8,0,8,-1};

  static int r=0;
  static int c = 0;


      #pragma HLS pipeline II=1
      // read out values from current line buffer
      for (int i = 0; i < 4; i ++ )
        smallbuf[i] = buf[i+1][c];
      // the new value is either 0 or read from frame
      if (r<MAX_HEIGHT && c<MAX_WIDTH)
      {
    	bit32 buf;
    	buf = Input_1.read();
        smallbuf[4] = (pixel_t)(((input_t)(buf(23, 16)) >> 8));
      }
      else if (c < MAX_WIDTH)
        smallbuf[4] = 0;
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
        pixel_t out_tmp;
        out_tmp = x_grad/12;
        Output_1.write(out_tmp.range(31,0));
        //gradient_y[r-2][c-2] = y_grad/12;
        out_tmp = y_grad/12;
        Output_2.write(out_tmp.range(31,0));
      }
      else if(r>=2 && c>=2)
      {
        //gradient_x[r-2][c-2] = 0;
        Output_1.write(0);
        //gradient_y[r-2][c-2] = 0;
        Output_2.write(0);
      }
      c++;
      if(c==MAX_WIDTH+2)
      {
    	  c=0;
    	  r++;
    	  if(r==MAX_HEIGHT+2)
    	  {
    		  r=0;
    	  }
      }
}

#define BUFFER_SIZE 3000
// calculate gradient in the z direction
void gradient_z_calc(
		hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Input_2,
		hls::stream<bit32> & Output_1
)
{
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Input_2
#pragma HLS INTERFACE ap_hs port=Output_1

  const int GRAD_WEIGHTS[] =  {1,-8,0,8,-1};
  static int r = 0;
  static int c = 0;
  static pixel_t ring_buf[BUFFER_SIZE];
  static int read_ptr = 0;
  static int write_ptr = 0;

      #pragma HLS pipeline II=1
  	  if(r<MAX_HEIGHT && c<MAX_WIDTH){
		  bit32 buf;
		  buf = Input_1.read();
		  pixel_t in1 = (pixel_t)((input_t)(buf(7 ,  0)) >> 8);
		  pixel_t in2 = ((input_t)(buf(15,  8)) >> 8);
		  pixel_t in3 = ((input_t)(buf(23, 16)) >> 8);
		  pixel_t in4 = ((input_t)(buf(31, 24)) >> 8);
		  buf = Input_2.read();
		  pixel_t in5 = ((input_t)(buf(7, 0)) >> 8);
		  pixel_t out_tmp;
		  out_tmp =((pixel_t)(in1*GRAD_WEIGHTS[0]
							+ in2*GRAD_WEIGHTS[1]
							+ in3*GRAD_WEIGHTS[2]
							+ in4*GRAD_WEIGHTS[3]
							+ in5*GRAD_WEIGHTS[4]))/12;
          ring_buf[write_ptr] = out_tmp;
          write_ptr++;
          if(write_ptr == BUFFER_SIZE) write_ptr=0;
  	  }

  	  if(r>=2 && c>=2){
  		  Output_1.write(ring_buf[read_ptr].range(31,0));
  		  read_ptr++;
  		if(read_ptr == BUFFER_SIZE) read_ptr=0;
  	  }

	  c++;
	  if(c==MAX_WIDTH+2)
	  {
	    c=0;
	    r++;
	    if(r==MAX_HEIGHT+2)
	    {
	  	  r=0;
	    }
	  }
}

// average the gradient in y direction
void gradient_weight_y(
		hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Input_2,
		hls::stream<bit32> & Input_3,
		hls::stream<bit32> & Output_1
   // gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH]
)
{
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Input_2
#pragma HLS INTERFACE ap_hs port=Input_3
#pragma HLS INTERFACE ap_hs port=Output_1


  hls::LineBuffer<7,MAX_WIDTH,gradient_t> buf;

  const pixel_t GRAD_FILTER[] = {0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755};
  static int r = 0;
  static int c = 0;

      #pragma HLS pipeline II=1
      #pragma HLS dependence variable=buf inter false

      if(r<MAX_HEIGHT)
      {
        buf.shift_pixels_up(c);
        gradient_t tmp;
        pixel_t in_tmp;
        tmp.x.range(31,0) = Input_1.read();
        tmp.y.range(31,0) = Input_2.read();
        tmp.z.range(31,0) = Input_3.read();
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
        Output_1.write(acc.x(31,0));
        Output_1.write(acc.y(31,0));
        Output_1.write(acc.z(31,0));
        //filt_grad[r-3][c] = acc;
      }
      else if(r>=3)
      {
        //filt_grad[r-3][c] = acc;
        Output_1.write(acc.x(31,0));
	    Output_1.write(acc.y(31,0));
	    Output_1.write(acc.z(31,0));
      }
      c++;
      if(c==MAX_WIDTH)
	  {
		c=0;
		r++;
		if(r==MAX_HEIGHT+3)
		{
		  r=0;
		}
	  }

}

// average gradient in the x direction
void gradient_weight_x(
		hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Output_1
)
		{
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
  hls::Window<1,7,gradient_t> buf;
  const pixel_t GRAD_FILTER[] = {0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755};
  static int r = 0;
  static int c = 0;

      #pragma HLS pipeline II=1
      buf.shift_pixels_left();
      gradient_t tmp;
      if(c<MAX_WIDTH)
      {
    	bit32 in_tmp;
    	in_tmp.range(31,0) = Input_1.read();
        tmp.x.range(31,0) = in_tmp.range(31,0);
    	in_tmp.range(31,0) = Input_1.read();
        tmp.y.range(31,0) = in_tmp.range(31,0);
    	in_tmp.range(31,0) = Input_1.read();
        tmp.z.range(31,0) = in_tmp.range(31,0);
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
        Output_1.write(acc.x(31,0));
		Output_1.write(acc.y(31,0));
		Output_1.write(acc.z(31,0));
      }
      else if(c>=3)
      {
        Output_1.write(acc.x(31,0));
  		Output_1.write(acc.y(31,0));
  		Output_1.write(acc.z(31,0));
      }

      c++;
	  if(c==MAX_WIDTH+3)
	  {
	    c=0;
	    r++;
	    if(r==MAX_HEIGHT)
	    {
	      r=0;
	    }
	  }

}

// outer product 
void outer_product(
		hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Output_1
		//gradient_t gradient[MAX_HEIGHT][MAX_WIDTH],
     //outer_t outer_product[MAX_HEIGHT][MAX_WIDTH]
		)
{
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
  static int r = 0;
  static int c = 0;

      #pragma HLS pipeline II=1
      gradient_t grad;
      grad.x.range(31,0)= Input_1.read();
      grad.y.range(31,0)= Input_1.read();
      grad.z.range(31,0)= Input_1.read();
      outer_pixel_t x = (outer_pixel_t) grad.x;
      outer_pixel_t y = (outer_pixel_t) grad.y;
      outer_pixel_t z = (outer_pixel_t) grad.z;
      outer_t out;
      out.val[0] = (x*x);
      out.val[1] = (y*y);
      out.val[2] = (z*z);
      out.val[3] = (x*y);
      out.val[4] = (x*z);
      out.val[5] = (y*z);
      bit32 out_tmp;
      out_tmp.range(31,0) = out.val[0].range(31,0);
      Output_1.write(out_tmp);
      out_tmp.range(15,0) = out.val[0].range(47,32);
      out_tmp.range(31,16) = out.val[1].range(15,0);
      Output_1.write(out_tmp);
      out_tmp.range(31,0) = out.val[1].range(47,16);
      Output_1.write(out_tmp);

      out_tmp.range(31,0) = out.val[2].range(31,0);
      Output_1.write(out_tmp);
      out_tmp.range(15,0) = out.val[2].range(47,32);
      out_tmp.range(31,16) = out.val[3].range(15,0);
      Output_1.write(out_tmp);
      out_tmp.range(31,0) = out.val[3].range(47,16);
      Output_1.write(out_tmp);
      out_tmp.range(31,0) = out.val[4].range(31,0);
      Output_1.write(out_tmp);
      out_tmp.range(15,0) = out.val[4].range(47,32);
      out_tmp.range(31,16) = out.val[5].range(15,0);
      Output_1.write(out_tmp);
      out_tmp.range(31,0) = out.val[5].range(47,16);
      Output_1.write(out_tmp);

      c++;
	  if(c==MAX_WIDTH)
	  {
		c=0;
		r++;
		if(r==MAX_HEIGHT)
		{
		  r=0;
		}
	  }
}

// tensor weight
void tensor_weight_y(//outer_t outer[MAX_HEIGHT][MAX_WIDTH],
		hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Output_1
                     //tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH]
												   )
{
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
  hls::LineBuffer<3,MAX_WIDTH,outer_t> buf;
  const pixel_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  static int r = 0;
  static int c = 0;
      #pragma HLS pipeline II=1
      
      outer_t tmp;
      #pragma HLS data_pack variable=tmp
      #pragma HLS data_pack variable=buf.val[0]
      buf.shift_pixels_up(c);
      if(r<MAX_HEIGHT)
      {
        //tmp = outer[r][c];
        bit32 buf;
        buf = Input_1.read();
        tmp.val[0].range(31,0) = buf.range(31,0);
        buf = Input_1.read();
        tmp.val[0].range(47,32) = buf.range(15,0);
        tmp.val[1].range(15,0)  = buf.range(31,16);
        buf = Input_1.read();
        tmp.val[1].range(47,16) = buf.range(31,0);

        buf = Input_1.read();
        tmp.val[2].range(31,0) = buf.range(31,0);
        buf = Input_1.read();
        tmp.val[2].range(47,32) = buf.range(15,0);
        tmp.val[3].range(15,0)  = buf.range(31,16);
        buf = Input_1.read();
        tmp.val[3].range(47,16) = buf.range(31,0);

        buf = Input_1.read();
        tmp.val[4].range(31,0) = buf.range(31,0);
        buf = Input_1.read();
        tmp.val[4].range(47,32) = buf.range(15,0);
        tmp.val[5].range(15,0)  = buf.range(31,16);
        buf = Input_1.read();
        tmp.val[5].range(47,16) = buf.range(31,0);
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
        bit32 out_tmp;
        out_tmp.range(31,0) = acc.val[0].range(31,0);
        Output_1.write(out_tmp);
        out_tmp.range(15,0) = acc.val[0].range(47,32);
        out_tmp.range(31,16) = acc.val[1].range(15,0);
        Output_1.write(out_tmp);
        out_tmp.range(31,0) = acc.val[1].range(47,16);
        Output_1.write(out_tmp);

        out_tmp.range(31,0) = acc.val[2].range(31,0);
        Output_1.write(out_tmp);
        out_tmp.range(15,0) = acc.val[2].range(47,32);
        out_tmp.range(31,16) = acc.val[3].range(15,0);
        Output_1.write(out_tmp);
        out_tmp.range(31,0) = acc.val[3].range(47,16);
        Output_1.write(out_tmp);

        out_tmp.range(31,0) = acc.val[4].range(31,0);
        Output_1.write(out_tmp);
        out_tmp.range(15,0) = acc.val[4].range(47,32);
        out_tmp.range(31,16) = acc.val[5].range(15,0);
        Output_1.write(out_tmp);
        out_tmp.range(31,0) = acc.val[5].range(47,16);
        Output_1.write(out_tmp);

      }
      c++;
	  if(c==MAX_WIDTH)
	  {
		c=0;
		r++;
		if(r==MAX_HEIGHT+1)
		{
		  r=0;
		}
	  }
}

void tensor_weight_x(
		hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Output_1
		//tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH],
                     //tensor_t tensor[MAX_HEIGHT][MAX_WIDTH]
		)
{
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
  hls::Window<1,3,tensor_t> buf;
  const pixel_t TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  //const float TENSOR_FILTER[] = {0.3243, 0.3513, 0.3243};
  static int r = 0;
  static int c = 0;


      #pragma HLS pipeline II=1
      buf.shift_pixels_left();
      tensor_t tmp;
      if(c<MAX_WIDTH)
      {
        //tmp = tensor_y[r][c];
        bit32 buf;
        buf = Input_1.read();
        tmp.val[0].range(31,0) = buf.range(31,0);
        buf = Input_1.read();
        tmp.val[0].range(47,32) = buf.range(15,0);
        tmp.val[1].range(15,0)  = buf.range(31,16);
        buf = Input_1.read();
        tmp.val[1].range(47,16) = buf.range(31,0);

        buf = Input_1.read();
        tmp.val[2].range(31,0) = buf.range(31,0);
        buf = Input_1.read();
        tmp.val[2].range(47,32) = buf.range(15,0);
        tmp.val[3].range(15,0)  = buf.range(31,16);
        buf = Input_1.read();
        tmp.val[3].range(47,16) = buf.range(31,0);

        buf = Input_1.read();
        tmp.val[4].range(31,0) = buf.range(31,0);
        buf = Input_1.read();
        tmp.val[4].range(47,32) = buf.range(15,0);
        tmp.val[5].range(15,0)  = buf.range(31,16);
        buf = Input_1.read();
        tmp.val[5].range(47,16) = buf.range(31,0);
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
        //tensor[r][c-1] = acc;
        bit32 out_tmp;
        out_tmp.range(31,0) = acc.val[0].range(31,0);
        Output_1.write(out_tmp);
        out_tmp.range(15,0) = acc.val[0].range(47,32);
        out_tmp.range(31,16) = acc.val[1].range(15,0);
        Output_1.write(out_tmp);
        out_tmp.range(31,0) = acc.val[1].range(47,16);
        Output_1.write(out_tmp);

        out_tmp.range(31,0) = acc.val[2].range(31,0);
        Output_1.write(out_tmp);
        out_tmp.range(15,0) = acc.val[2].range(47,32);
        out_tmp.range(31,16) = acc.val[3].range(15,0);
        Output_1.write(out_tmp);
        out_tmp.range(31,0) = acc.val[3].range(47,16);
        Output_1.write(out_tmp);

        out_tmp.range(31,0) = acc.val[4].range(31,0);
        Output_1.write(out_tmp);
        out_tmp.range(15,0) = acc.val[4].range(47,32);
        out_tmp.range(31,16) = acc.val[5].range(15,0);
        Output_1.write(out_tmp);
        out_tmp.range(31,0) = acc.val[5].range(47,16);
        Output_1.write(out_tmp);

      }
      c++;
	  if(c==MAX_WIDTH+1)
	  {
		c=0;
		r++;
		if(r==MAX_HEIGHT)
		{
		  r=0;
		}
	  }
}

// compute output flow
void flow_calc(
		hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Output_1
		)
{
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
  static outer_pixel_t buf[2];
  static int r = 0;
  static int c = 0;

      #pragma HLS pipeline II=1
      tensor_t tmp_tensor;// = tensors[r][c];
      bit32 in_tmp;
      in_tmp = Input_1.read();
      tmp_tensor.val[0].range(31,0) = in_tmp.range(31,0);
      in_tmp = Input_1.read();
      tmp_tensor.val[0].range(47,32) = in_tmp.range(15,0);
      tmp_tensor.val[1].range(15,0)  = in_tmp.range(31,16);
      in_tmp = Input_1.read();
      tmp_tensor.val[1].range(47,16) = in_tmp.range(31,0);

      in_tmp = Input_1.read();
      tmp_tensor.val[2].range(31,0) = in_tmp.range(31,0);
      in_tmp = Input_1.read();
      tmp_tensor.val[2].range(47,32) = in_tmp.range(15,0);
      tmp_tensor.val[3].range(15,0)  = in_tmp.range(31,16);
      in_tmp = Input_1.read();
      tmp_tensor.val[3].range(47,16) = in_tmp.range(31,0);

      in_tmp = Input_1.read();
      tmp_tensor.val[4].range(31,0) = in_tmp.range(31,0);
      in_tmp = Input_1.read();
      tmp_tensor.val[4].range(47,32) = in_tmp.range(15,0);
      tmp_tensor.val[5].range(15,0)  = in_tmp.range(31,16);
      in_tmp = Input_1.read();
      tmp_tensor.val[5].range(47,16) = in_tmp.range(31,0);
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

      vel_pixel_t out_tmp;
      out_tmp = (vel_pixel_t)buf[0];
      Output_1.write(out_tmp.range(31,0));
      out_tmp = (vel_pixel_t)buf[1];
      Output_1.write(out_tmp.range(31,0));

      c++;
	  if(c==MAX_WIDTH)
	  {
		c=0;
		r++;
		if(r==MAX_HEIGHT)
		{
		  r=0;
		}
	  }
}

void unpack(
		hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Output_1,
		hls::stream<bit32> & Output_2,
		hls::stream<bit32> & Output_3
		)
{
#pragma HLS INTERFACE ap_hs port=Input_1
#pragma HLS INTERFACE ap_hs port=Output_1
#pragma HLS INTERFACE ap_hs port=Output_2
#pragma HLS INTERFACE ap_hs port=Output_3
	bit32 buf;
	buf = Input_1.read();
	Output_1.write(buf(31,0));
	Output_2.write(buf(31,0));
	buf = Input_1.read();
	Output_3.write(buf(31,0));

}
// top-level kernel function
void optical_flow(
		hls::stream<bit32> & Input_1,
		hls::stream<bit32> & Output_1
                  //velocity_t outputs[MAX_HEIGHT][MAX_WIDTH]
		)
{
  hls::stream<ap_uint<32> > unpack_out1("sb1");
  hls::stream<ap_uint<32> > unpack_out2("sb2");
  hls::stream<ap_uint<32> > unpack_out3("sb3");
  hls::stream<ap_uint<32> > gradient_xy_calc_out1("sb4");
  hls::stream<ap_uint<32> > gradient_xy_calc_out2("sb5");
  hls::stream<ap_uint<32> > gradient_z_calc_out1("sb6");
  hls::stream<ap_uint<32> > gradient_weight_y_out1("sb7");
  hls::stream<ap_uint<32> > gradient_weight_x_out1("sb8");
  hls::stream<ap_uint<32> > outer_product_out1("sb9");
  hls::stream<ap_uint<32> > tensor_weight_y_out1("sb10");
  hls::stream<ap_uint<32> > tensor_weight_x_out1("sb11");
  #pragma HLS DATAFLOW

  // FIFOs connecting the stages
  static pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH];



  // stream in and organize the inputs
  static bit32 buf;
  int r, c;

  for (r=0; r<MAX_HEIGHT+6; r++)
  {
	printf ("r=%d\n", r);
    for (c=0; c<MAX_WIDTH+6; c++)
    {
    	if((r<MAX_HEIGHT) && (c<MAX_WIDTH))
		{
			unpack(Input_1, unpack_out1, unpack_out2, unpack_out3);

		}
    	if((r<MAX_HEIGHT+2) && (c<MAX_WIDTH+2))
    	{
    		gradient_z_calc(unpack_out2, unpack_out3, gradient_z_calc_out1);
    		gradient_xy_calc(unpack_out1, gradient_xy_calc_out1, gradient_xy_calc_out2);
    	}

    	if((r<MAX_HEIGHT+5) && (r>=2) && (c<MAX_WIDTH+2) && (c>=2))
		{
    	  gradient_weight_y(gradient_xy_calc_out1, gradient_xy_calc_out2, gradient_z_calc_out1, gradient_weight_y_out1);
		}
    	if((r<MAX_HEIGHT+5) && (r>=5) && (c<MAX_WIDTH+5) && (c>=2))
    	{
    		gradient_weight_x(gradient_weight_y_out1, gradient_weight_x_out1);

    	}
    	if((r<MAX_HEIGHT+5) && (r>=5) && (c<MAX_WIDTH+5) && (c>=5))
    	{
    		outer_product(gradient_weight_x_out1, outer_product_out1);
    	}

    	if((r<MAX_HEIGHT+6) && (r>=5) && (c<MAX_WIDTH+5) && (c>=5))
		{
    		tensor_weight_y(outer_product_out1, tensor_weight_y_out1);
		}

    	if((r<MAX_HEIGHT+6) && (r>=6) && (c<MAX_WIDTH+6) && (c>=5))
		{
    		tensor_weight_x(tensor_weight_y_out1, tensor_weight_x_out1);
		}

    	if((r<MAX_HEIGHT+6) && (r>=6) && (c<MAX_WIDTH+6) && (c>=6))
		{
			flow_calc(tensor_weight_x_out1, Output_1);
		}
    }
  }


  //
  // compute








}

/*
void data_gen(
		hls::stream<bit32> & Output_1
		)
{
#pragma HLS INTERFACE ap_hs port=Output_1

	for (int i=0; i<1024; i++)
	{
#pragma HLS PIPELINE II=1
		Output_1.write(i);
	}
}

*/
