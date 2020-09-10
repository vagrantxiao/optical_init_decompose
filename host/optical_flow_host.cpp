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
#define SDSOC


#include <sys/time.h>


#ifdef SDSOC
  // sdsoc headers
  //#include "sds_lib.h"
  // hardware function declaration
  #include "../sdsoc/optical_flow.h"
#endif

// here we use an image library to handle file IO

// other headers
#include "typedefs.h"
#include "input_data.h"



void check_results(velocity_t output[MAX_HEIGHT][MAX_WIDTH])
{
  double sum = 0;
  // copy the output into the float image
  for (int i = 0; i < MAX_HEIGHT; i++)
  {
    for (int j = 0; j < MAX_WIDTH; j++)
    {

        double out_x = output[i][j].x;
        double out_y = output[i][j].y;
        //printf("out_x=%f, out_y=%f\n", out_x, out_y);
        sum += (out_x+out_y);
    }
  }


  double avg_error = sum / (MAX_HEIGHT*MAX_WIDTH);
  printf("Correct Average error: -0.625107 degrees\n");
  printf("Average error: %lf degrees\n", avg_error);

}


int main(int argc, char ** argv) 
{
  hls::stream<frames_t> Input_1;


  printf("Optical Flow Application\n");
  struct timeval start, end;
  // sdsoc version host code
  #ifdef SDSOC
    // input and output buffers
    frames_t frames[MAX_HEIGHT][MAX_WIDTH];
    velocity_t outputs[MAX_HEIGHT][MAX_WIDTH];

    // pack the values
    for (int i = 0; i < MAX_HEIGHT; i++) 
    {
      for (int j = 0; j < MAX_WIDTH; j++) 
      {
        frames[i][j](31 ,  0) = input_data[(MAX_WIDTH*i+j)%10240*2+1];
        frames[i][j](63,  32) = input_data[(MAX_WIDTH*i+j)%10240*2];
        //Input_1.write(frames[i][j]);
      }
    }

    // run
    gettimeofday(&start, NULL);
    optical_flow(frames, outputs);
    gettimeofday(&end, NULL);

  #endif

  // check results
  printf("Checking results:\n");

  check_results(outputs);

  // print time
  long long elapsed = (end.tv_sec - start.tv_sec) * 1000000LL + end.tv_usec - start.tv_usec;   
  printf("elapsed time: %lld us\n", elapsed);


  return EXIT_SUCCESS;

}
