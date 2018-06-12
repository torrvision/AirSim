 // Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "common/common_utils/StrictMode.hpp"
STRICT_MODE_OFF
#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif // !RPCLIB_MSGPACK
#include "rpc/rpc_error.h"
STRICT_MODE_ON

#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include "common/common_utils/FileSystem.hpp"
#include <iostream>
#include <chrono> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
#include <pthread.h>

using namespace msr::airlib;

typedef ImageCaptureBase::ImageRequest ImageRequest;
typedef ImageCaptureBase::ImageResponse ImageResponse;
typedef ImageCaptureBase::ImageType ImageType;
typedef common_utils::FileSystem FileSystem;

std::map<std::string,cv::Mat> pullImage(bool, std::string);
void showImage(cv::Mat, std::string, int, bool);
cv::Mat pullDepthImage(bool, int, std::string);

struct threadArgs
{
	msr::airlib::MultirotorRpcLibClient* client;
	double frameRate;
	int saveType;
};

/*
  ++++++++++++++++++++++++++++++++++++++ ERRORS +++++++++++++++++++++++++++++++++++++++++++++++++++++++
  * For some reason cv::namedWindow, cv::imshow and cv::waitKey all cause 'Error: free(): invalid pointer'
  * even when function isn't called
  * 
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
*/

void showImage(cv::Mat img, std::string windowName, int wKey =  30, bool normalise = true)
{
    //normalise image
    if(normalise)
    {
		cv::normalize(img,img,0,255,cv::NORM_MINMAX);
    }
	cv::imshow(windowName,img);
	cv::waitKey(wKey);
}


//function to query simulator for depth images, converting to cv::Mat and saving
//input: save images? save type and path to save directory
//outut: cv::Mat image of format CV_32FC1
cv::Mat pullDepthImage(bool saveImages, msr::airlib::MultirotorRpcLibClient *client, int saveType = 0, std::string path="/home/nvidia/Documents/AirSimImages/")
{
    //Set request to pull one scene and one depth image
    vector<ImageRequest> request = {ImageRequest(1, ImageType::DepthVis, true) };
    const vector<ImageResponse>& response = client->simGetImages(request);

    //pull failed
    if(response.size() <= 0)
    {
        std::cerr << "No images pulled" << std::endl;
        throw std::exception ();
    }

    ImageResponse image_info = response.front();

    cv::Mat depth(image_info.width, image_info.height, CV_32FC1);

    for(int i = 0; i < image_info.width*image_info.height; i++)
    {
        depth.at<float>( i % image_info.width,(int) i / image_info.width) = image_info.image_data_float.at(i);
    }
    
    depth = depth.t();
    
    if(saveImages)
    {
		std::string file_path = FileSystem::combine(path, std::to_string(image_info.time_stamp));
		if(saveType == 0)
		{
			imwrite(file_path+".png",depth);
		}
		else if(saveType == 1)
		{
			Utils::writePfmFile(image_info.image_data_float.data(), image_info.width, image_info.height, file_path + ".pfm");
		}        
    }

    //transpose depth
    return depth;
}

//function to engage drone API and initiate flight plan
//input: -
//output: -
void *flightPlan(void * cli)
{
	msr::airlib::MultirotorRpcLibClient *client = (msr::airlib::MultirotorRpcLibClient *) cli;
	//comment out to use remote controller
	client->enableApiControl(true);
	client->armDisarm(true);
        
	float takeoffTimeout = 1; 
	client->takeoff(takeoffTimeout);

	// switch to explicit hover mode so that this is the fallback when 
	// move* commands are finished.
	std::this_thread::sleep_for(std::chrono::duration<double>(1));
	client->hover();
	
	//Rotate drone 90 degrees clockwise
	client->rotateToYaw(90.0f,3);
	std::this_thread::sleep_for(std::chrono::duration<double>(3));

	// moveByVelocityZ is an offboard operation, so we need to set offboard mode.
	client->enableApiControl(true); 
	auto position = client->getPosition();
	float z = position.z(); // current position (NED coordinate system).  
	const float speed = 3.0f;
	const float size = 10.0f; 
	const float duration = size / speed;
	DrivetrainType driveTrain = DrivetrainType::ForwardOnly;
	YawMode yaw_mode(true, 0);
	
	std::cout << "moveByVelocityZ(" << speed << ", 0, " << z << "," << duration << ")" << std::endl;
	client->moveByVelocityZ(speed, 0, z, duration, driveTrain, yaw_mode);
	std::this_thread::sleep_for(std::chrono::duration<double>(duration));
	
	std::cout << "moveByVelocityZ(0, " << speed << "," << z << "," << duration << ")" << std::endl;
	client->moveByVelocityZ(0, speed, z, duration, driveTrain, yaw_mode);
	std::this_thread::sleep_for(std::chrono::duration<double>(duration));
	
	std::cout << "moveByVelocityZ(" << -speed << ", 0, " << z << "," << duration << ")" << std::endl;
	client->moveByVelocityZ(-speed, 0, z, duration, driveTrain, yaw_mode);
	std::this_thread::sleep_for(std::chrono::duration<double>(duration));
	
	std::cout << "moveByVelocityZ(0, " << -speed << "," << z << "," << duration << ")" << std::endl;
	client->moveByVelocityZ(0, -speed, z, duration, driveTrain, yaw_mode);
	std::this_thread::sleep_for(std::chrono::duration<double>(duration));
	
	
	client->hover();
	client->land();
	std::this_thread::sleep_for(std::chrono::duration<double>(3.0f));
	
	client->armDisarm(false);
	client->reset();
	client->enableApiControl(false);
	//pthread_exit(NULL);
}

//function to save pull and save depth images at 0.5Hz
//input: struct containing frame rate and extension for saving
//output: -
void *saveImages(void *threadData)
{
	struct threadArgs *data;
	data = (struct threadArgs *) threadData;
	
	cv::Mat depth;
	double duration = 1/ data->frameRate;
	while(true)
	{
		depth = pullDepthImage(true, data->client, data->saveType);
		showImage(depth, "test", 30, true);
		//std::this_thread::sleep_for(std::chrono::duration<double>(duration));
	}
}


int main() 
{
 
    try {
		msr::airlib::MultirotorRpcLibClient client("172.16.0.1");
        client.confirmConnection();
        
        //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        //create thread to save depth images
        pthread_t depthThread;
        struct threadArgs threadData;
        threadData.client = &client;
        threadData.frameRate = 0.5;
        threadData.saveType = 1;
        int rc = pthread_create(&depthThread, NULL, saveImages, (void *)&threadData);
        
        if(rc)
        {
			std::cerr << "Error: unable to create saveImages() thread" << std::endl;
		}
		//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		
		//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        //create thread to initiate flight plan
        pthread_t flightThread;
        rc = pthread_create(&flightThread, NULL, flightPlan, (void *) &client);
        //set thread to block main until completion
        pthread_join(flightThread,nullptr);
        
        if(rc)
        {
			std::cerr << "Error: unable to create saveImages() thread" << std::endl;
		}
		//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		

    }
    catch (rpc::rpc_error&  e) {
        std::string msg = e.get_error().as<std::string>();
        std::cout << "Exception raised by the API, something went wrong." << std::endl << msg << std::endl;
    }
	
	return 0;
}
