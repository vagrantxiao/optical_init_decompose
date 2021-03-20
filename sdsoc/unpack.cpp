#include "../host/typedefs.h"

void unpack(
		hls::stream<frames_t> & Input_1,
		hls::stream< bit32 > & Output_1,
		hls::stream< bit32 > & Output_2,
		hls::stream< bit32 > & Output_3,
		hls::stream< bit32 > & Output_4,
		hls::stream< bit32 > & Output_5,
		hls::stream< bit32 > & Output_6
									 )
{

	static frames_t buf;
	input_t frame1_a, frame2_a, frame3_a, frame4_a, frame5_a, frame3_b;
	bit32 out_tmp;
	out_tmp = 0;
	FRAMES_CP_OUTER: for (int r=0; r<MAX_HEIGHT; r++)
	  {
		FRAMES_CP_INNER: for (int c=0; c<MAX_WIDTH; c++)
		{
		  #pragma HLS pipeline II=1

		  // one wide read
		  buf = Input_1.read();
		  // printf("0x%08x\n",(unsigned int) buf(63, 32));
		  // printf("0x%08x\n",(unsigned int) buf(31,  0));
		  // assign values to the FIFOs


		  frame1_a = ((input_t)(buf(7 ,  0)) >> 8);
		  out_tmp(16, 0) = frame1_a(16, 0);
		  Output_1.write(out_tmp);


		  frame2_a = ((input_t)(buf(15,  8)) >> 8);
		  out_tmp(16, 0) = frame2_a(16, 0);
		  Output_2.write(out_tmp);


		  frame3_a = ((input_t)(buf(23, 16)) >> 8);
		  out_tmp(16, 0) = frame3_a(16, 0);
		  Output_5.write(out_tmp);


		  frame3_b = ((input_t)(buf(23, 16)) >> 8);
		  out_tmp(16, 0) = frame3_b(16, 0);
		  Output_6.write(out_tmp);


		  frame4_a = ((input_t)(buf(31, 24)) >> 8);
		  out_tmp(16, 0) = frame4_a(16, 0);
		  Output_3.write(out_tmp);


		  frame5_a = ((input_t)(buf(39, 32)) >> 8);
		  out_tmp(16, 0) = frame5_a(16, 0);
		  Output_4.write(out_tmp);

		}
	  }

}
