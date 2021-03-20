#include "../host/typedefs.h"


// calculate gradient in the z direction
void gradient_z_calc(
	hls::stream< bit32> & Input_1,
	hls::stream< bit32 > & Input_2,
	hls::stream< bit32 > & Input_3,
	hls::stream< bit32 > & Input_4,
	hls::stream< bit32 > & Input_5,
	hls::stream< bit32 > & Output_1
	)
{

	input_t frame1, frame2, frame3, frame4, frame5;
	bit32 in1_tmp, in2_tmp, in3_tmp, in4_tmp, in5_tmp;
	pixel_t gradient_z;
	bit32 out_tmp;

  const int GRAD_WEIGHTS[] =  {1,-8,0,8,-1};
  GRAD_Z_OUTER: for(int r=0; r<MAX_HEIGHT; r++)
  {
    GRAD_Z_INNER: for(int c=0; c<MAX_WIDTH; c++)
    {
      #pragma HLS pipeline II=1
      in1_tmp = Input_1.read();
      frame1(16, 0) = in1_tmp(16, 0);
      in2_tmp = Input_2.read();
      frame2(16, 0) = in2_tmp(16, 0);
      in3_tmp = Input_3.read();
      frame3(16, 0) = in3_tmp(16, 0);
      in4_tmp = Input_4.read();
      frame4(16, 0) = in4_tmp(16, 0);
      in5_tmp = Input_5.read();
      frame5(16, 0) = in5_tmp(16, 0);
      gradient_z =((pixel_t)(frame1*GRAD_WEIGHTS[0]
                        + frame2*GRAD_WEIGHTS[1]
                        + frame3*GRAD_WEIGHTS[2]
                        + frame4*GRAD_WEIGHTS[3]
                        + frame5*GRAD_WEIGHTS[4]))/12;
      out_tmp(31, 0) = gradient_z(31, 0);
      Output_1.write(out_tmp);
    }
  }
}
