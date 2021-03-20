/*===============================================================*/
/*                                                               */
/*                    optical_flow_host.cpp                      */
/*                                                               */
/*      Main host function for the Optical Flow application.     */
/*                                                               */
/*===============================================================*/

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <time.h>
#include <sys/time.h>
#include "utils.h"
#include "typedefs.h"
#include "check_result.h"
#include "../sdsoc/optical_flow.h"


void data_gen(
		hls::stream< frames_t > &Output_1)
{
#pragma HLS interface ap_hs port=Output_1


	#include "./input_data.h"
	#pragma HLS ARRAY_PARTITION variable=input_data cyclic factor=2 dim=1

	int i;
	for (i=0; i<446464; i++)
	{
#pragma HLS pipeline II=2
		frames_t tmp;
		tmp(63,32) = input_data[i*2];
		//tmp(63,32) = 0;
		tmp(31, 0) = input_data[i*2+1];

		Output_1.write(tmp);
		//Output_1.write(tmp(95,64));
		//Output_1.write(tmp(127,96));
	}
}



int main(int argc, char ** argv) 
{
  printf("Optical Flow Application\n");

  // parse command line arguments
  std::string dataPath("");
  std::string outFile("");

  // for sw and sdsoc versions
  parse_sdsoc_command_line_args(argc, argv, dataPath, outFile);

  // create actual file names according to the datapath
  std::string frame_files[5];
  std::string reference_file;
  frame_files[0] = dataPath + "/frame1.ppm";
  frame_files[1] = dataPath + "/frame2.ppm";
  frame_files[2] = dataPath + "/frame3.ppm";
  frame_files[3] = dataPath + "/frame4.ppm";
  frame_files[4] = dataPath + "/frame5.ppm";
  reference_file = dataPath + "/ref.flo";

  // read in images and convert to grayscale
  printf("Reading input files ... \n");

  CByteImage imgs[5];
  for (int i = 0; i < 5; i++) 
  {
    CByteImage tmpImg;
    ReadImage(tmpImg, frame_files[i].c_str());
    imgs[i] = ConvertToGray(tmpImg);
  }

  // read in reference flow file
  printf("Reading reference output flow... \n");

  CFloatImage refFlow;
  ReadFlowFile(refFlow, reference_file.c_str());

  // timers
  struct timeval start, end;

  // sdsoc version host code
    // input and output buffers
    //static frames_t frames[MAX_HEIGHT][MAX_WIDTH];
    static velocity_t outputs[MAX_HEIGHT][MAX_WIDTH];


    ap_uint<128>  tmpframes;
    static hls::stream< frames_t > frames("test1");
    static hls::stream< ap_uint<32> > flo_out("test2");

    data_gen(frames);
    printf("Start!\n");

    // run
    gettimeofday(&start, NULL);
    optical_flow(frames, outputs);
    printf("Almost there!/n");
    gettimeofday(&end, NULL);


  // check results
  printf("Checking results:\n");
  printf("The right Average error should be 32.058417\n");
  check_results(outputs, refFlow, outFile);

  // print time
  long long elapsed = (end.tv_sec - start.tv_sec) * 1000000LL + end.tv_usec - start.tv_usec;   
  printf("elapsed time: %lld us\n", elapsed);


  return EXIT_SUCCESS;

}
