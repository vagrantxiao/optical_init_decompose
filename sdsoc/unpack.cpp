#include "../host/typedefs.h"

void unpack(
		hls::stream<frames_t> & Input_1,
		input_t frame1_a[MAX_HEIGHT][MAX_WIDTH],
		input_t frame2_a[MAX_HEIGHT][MAX_WIDTH],
		input_t frame4_a[MAX_HEIGHT][MAX_WIDTH],
		input_t frame5_a[MAX_HEIGHT][MAX_WIDTH],
		input_t frame3_a[MAX_HEIGHT][MAX_WIDTH],
		input_t frame3_b[MAX_HEIGHT][MAX_WIDTH]
									 )
{

	static frames_t buf;

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
		  frame1_a[r][c] = ((input_t)(buf(7 ,  0)) >> 8);
		  frame2_a[r][c] = ((input_t)(buf(15,  8)) >> 8);
		  frame3_a[r][c] = ((input_t)(buf(23, 16)) >> 8);
		  frame3_b[r][c] = ((input_t)(buf(23, 16)) >> 8);
		  frame4_a[r][c] = ((input_t)(buf(31, 24)) >> 8);
		  frame5_a[r][c] = ((input_t)(buf(39, 32)) >> 8);
		}
	  }

}
