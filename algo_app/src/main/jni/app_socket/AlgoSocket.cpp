#include "AlgoSocket.h"
#include "ninebot_log.h"
#include "AlgoUtils.h"
#include "Quaternion.h"
#include "Vector3.h"

//#include <unistd.h>

#include <algorithm>

#define PI 3.1415

namespace ninebot_algo
{
	namespace elevator_simulation_algo
	{
		using namespace std;
		using namespace cv;

		AlgoElevator::~AlgoElevator() {
            this->stopPoseRecord();

            if(m_p_local_mapping) {
                delete m_p_local_mapping;
                m_p_local_mapping = NULL;
            }

            if(m_p_localmapping_rawdata != NULL) {
                delete m_p_localmapping_rawdata;
                m_p_localmapping_rawdata = NULL;
            }

            if(m_p_model_based_planner){
                delete m_p_model_based_planner;
                m_p_model_based_planner = NULL;
            }
		}

		bool AlgoElevator::init(test_params_t cParams)
		{
			raw_depth.image = cv::Mat::zeros(cv::Size(320, 240), CV_16UC1);
			raw_depth.timestampSys = 0;
			t_old = 0;
		    imgstream_en = true;
            nStep = 0;
            elevator_scenario_state = 0; // No functioning state

			mRawDataInterface->getMaincamParam(raw_camerapara);
			mRawDataInterface->getCalibrationDS4T(calib);

            string serial = RawData::retrieveRobotSerialNumber();
			ALOGD("robot serial: %s", serial.c_str());
			ALOGD("robot model: %d", RawData::retrieveRobotModel());
			m_is_init_succed = true;

            m_timestamp_start = mRawDataInterface->getCurrentTimestampSys();

            initLocalMapping();

            initLocalPlanner();

            mRawDataInterface->ExecuteHeadMode(0);
            mRawDataInterface->ExecuteHeadSmoothFactor(1,1,1);
            mRawDataInterface->ExecuteHeadPos(0, 0.7, 0);

			return m_is_init_succed;
		}


		void AlgoElevator::UpdateAccel(MTPoint val)
		{
			raw_accel = val;
		}

		void AlgoElevator::UpdateGyro(MTPoint val)
		{
			raw_gyro = val;
		}

		void AlgoElevator::toggleImgStream()
		{
			imgstream_en = !imgstream_en;
		}

		void AlgoElevator::setVLSopen(bool en)
        {
            if(en)
                mRawDataInterface->startVLS(false);
            else
                mRawDataInterface->stopVLS();
        }

        void AlgoElevator::startPoseRecord(std::string save_folder, int64_t save_time)
        {
            pose_isRecording = true;
            nStep = 0;
            m_folder_socket = "/sdcard/socket/";
            createFolder(m_folder_socket);        
            //m_state_file.open(m_folder_socket + "state.txt");

            ///// Save map to file /////

            cv::Mat fix_map;
            int height, width, sel;
            Eigen::Vector3d mapCenter;
            m_p_local_mapping->getFixMap( fix_map, mapCenter, width, height, sel=0);

            ALOGD("Writing to file");
            //string filename_map = m_folder_socket + "big_map_test" + ".bmp";
            //cv::imwrite(filename_map, show2);
            cv::Mat show = mapToShow(m_local_map);
            cv::resize(show, show, cv::Size(360,360));
            addShowFov(raw_odometry.twist.pose.orientation, 1.0, show, 0.05f);


            vector<int> compression_params;
            compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);

            try {
                cv::imwrite("/sdcard/socket/local_map_c.png", show, compression_params);
            }
            catch (runtime_error& ex) {
                ALOGD("Catch");
                fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
                //return 1;
            }
            ALOGD("End writing to file");
        }

        void AlgoElevator::stopPoseRecord()
        {
            pose_isRecording = false;
            //m_state_file.close();
        }

        void AlgoElevator::onEmergencyStop(bool is_stop)
        {
            m_is_stop = is_stop;
        }

        void AlgoElevator::onWheelSide(int event)
        {
            m_slide_event = event;
        }

        void AlgoElevator::startMotionTest()
        {
            motion_test = 1;
            motion_sign = -motion_sign;
        }

		bool AlgoElevator::step()
		{
			/*! Get start timestamp for calculating algorithm runtime */
			auto start = std::chrono::high_resolution_clock::now();

			if(!m_is_init_succed){
				ALOGD("AlgoSocket: init false");
				return false;
			}

            if(imgstream_en){
                mRawDataInterface->retrieveDepth(raw_depth, false);
                if(raw_depth.timestampSys==0)
                {
                    ALOGD("AlgoSocket: depth wrong");
                    return false;
                }
                // check if depth is duplicate
                if(t_old==raw_depth.timestampSys)
                {
                    ALOGD("fisheye duplicate: %lld",raw_depth.timestampSys/1000);
                    return true;
                }
                else
                {
                    t_old=raw_depth.timestampSys;
                } 
            }

            /*! **********************************************************************
             * **** Local Map on Robot
             * ********************************************************************* */         
            mRawDataInterface->retrieveOdometry(raw_odometry, -1);     
            mRawDataInterface->retrieveBaseVelocity(raw_velocity);
            prepare_localmap_and_pose_for_controller_g1();

            /*! **********************************************************************
             * **** Local Obstacle Avoidance on Robot 
             * ********************************************************************* */         
            if (m_has_path_init)
                this->stepLocalPlanner();

            /*! **********************************************************************
             * **** Local Detection on Robot
             * ********************************************************************* */                     
            detectPersonFromMap();


			/*! **********************************************************************
			 * **** Processing the algorithm with all input and sensor data **********
			 * ********************************************************************* */
			 string contents;

			 if(m_isRender)
			 {
				/*! Draw the result to canvas */
				cv::Mat tfisheyeS, tfisheye;
				canvas.setTo(240);

				if(imgstream_en){

                    cv::Mat tdepth = raw_depth.image / 10;
                    cv::Mat tdepth8, tdepthup, tdepth8color;
                    tdepth.convertTo(tdepth8, CV_8U);
                    resize(tdepth8, tdepthup, Size(), 0.75, 0.75);
                    cv::Mat ca1 = canvas(Rect(0, 0, 240, 180));
                    applyColorMap(tdepthup, tdepth8color, cv::COLORMAP_JET);
                    tdepth8color.copyTo(ca1);

                    if(!m_local_map.empty()){
                        cv::Mat show = mapToShow(m_local_map);
                        for (auto person : m_persons) {
                            cv::rectangle(show, cv::Point2f(person.first - 5, person.second - 5), cv::Point2f(person.first + 5, person.second + 5), cv::Scalar(255, 0, 0));
                        }
                        cv::resize(show, show, cv::Size(360,360)); 
                        addShowFov(raw_odometry.twist.pose.orientation, 1.0, show, 0.05f);

                        //// Show depth cam orientation fov, just to display on screen where the
                        // camera is looking (since orientation is inversed and can be confusing)

                        // merge all TF requests
                        ninebot_tf::vector_req_t reqList;
                        ninebot_tf::vector_tf_msg_t resList;
                        reqList.push_back(ninebot_tf::tf_request_message("base_center_ground_frame", "world_odom_frame", raw_depth.timestampSys, 500));
                        reqList.push_back(ninebot_tf::tf_request_message("rsdepth_center_neck_fix_frame", "world_odom_frame", raw_depth.timestampSys, 500));
                        mRawDataInterface->getMassiveTfData(&reqList,&resList);
                        if(resList.size()!=2){
                            ALOGE("getMassiveTfData wrong");
                            return false;
                        }

                        ninebot_tf::tf_message tf_msg = resList[0];
                        ninebot_tf::tf_message tf_msg2 = resList[1];
                        if(tf_msg.err==0 && tf_msg2.err==0){
                            //Show orientation of current field of view for the depth camera
                            addShowFov(tfmsgTo2DPose(tf_msg2).pose.orientation, 1.0, show, 0.05f);
                        }
                        else{
                            ALOGE("localmap: Could not find tf pose with specified depth ts, error code: %d, %d", tf_msg.err, tf_msg2.err);
                        }
                        ///// Show depth cam orientation fov end /////

                        ALOGD("show: size =(%zu,%zu)", show.rows, show.cols);
                        cv::Mat ca2 = canvas(cv::Rect(240, 0, show.cols, show.rows));
                        show.copyTo(ca2);
                    }

                    int type = raw_depth.image.type();
                    string r;
                    {
                      uchar depth = type & CV_MAT_DEPTH_MASK;
                      uchar chans = 1 + (type >> CV_CN_SHIFT);
                      switch ( depth ) {
                        case CV_8U:  r = "8U"; break;
                        case CV_8S:  r = "8S"; break;
                        case CV_16U: r = "16U"; break;
                        case CV_16S: r = "16S"; break;
                        case CV_32S: r = "32S"; break;
                        case CV_32F: r = "32F"; break;
                        case CV_64F: r = "64F"; break;
                        default:     r = "User"; break;
                      }
                      r += "C";
                      r += (chans+'0');
                    }            

                    if (m_safety_control){
                        contents = "Safety: TRUE"; 
                    }
                    else {
                        contents = "Safety: FALSE"; 
                    }
                    putText(canvas, contents, cv::Point(1, 280), CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0,0,0), 2);

                }

				/*! Copy internal canvas to intermediate buffer mDisplayIm */
                setDisplayData();
			}

			/*! Calculate the algorithm runtime */
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> elapsed = end-start;
			ALOGD("step time: %f",elapsed.count());
			{
				std::lock_guard<std::mutex> lock(mMutexTimer);
				m_ptime = elapsed.count()*0.5 + m_ptime*0.5;
			}

			return true;
		}


        void AlgoElevator::switchSafetyControl() {
            m_safety_control = !m_safety_control;
        }

        void AlgoElevator::safeControl(float v, float w) {

            if (m_safety_control){
                const float kCloseObstacleThres = 1.0;
                if(m_ultrasonic_average < kCloseObstacleThres*1000){
                    if(v>0 && std::abs(v/std::fmax(0.01,w)) > 0.8){
                        StampedVelocity velocity;
                        mRawDataInterface->retrieveBaseVelocity(velocity);
                        float v_emergency = std::min(0.0, 0.2-velocity.vel.linear_velocity);
                        ALOGE("command dangerous: (%f,%f), current vel: (%f,%f), ultrasonic_average = %f: v_emergency = %f", v, w, velocity.vel.linear_velocity, velocity.vel.angular_velocity, m_ultrasonic_average, v_emergency);
                        mRawDataInterface->ExecuteCmd(v_emergency, 0.0f, 0);

                        // Control the head yaw angle
                        if(v_emergency==0){
                            // If robot is not moving at all, look around (scan head)
                            StampedHeadPos raw_headpos;
                            mRawDataInterface->retrieveHeadPos(raw_headpos);
                            mRawDataInterface->ExecuteHeadMode(2);
                            scanHead(raw_headpos.pitch,110);

                            // If scenario is finished, return head to base position
                            if(elevator_scenario_state >= END){
                                mRawDataInterface->ExecuteHeadMode(0);
                                mRawDataInterface->ExecuteHeadPos(0,0,1);
                            }
                        }
                        else{
                            // If turning speed is 0, go to position mode and put head at normal angle to see in front
                            mRawDataInterface->ExecuteHeadMode(0);
                            mRawDataInterface->ExecuteHeadPos(0,0,1);
                        }

                        return;
                    }
                }
            }

            // Make robot base move
            mRawDataInterface->ExecuteCmd(v, w, 0);

            // Control robot head yaw angle based on current speed
            if(w==0){
                if(v==0){
                    // If robot is not moving at all, look around (scan head)
                    StampedHeadPos raw_headpos;
                    mRawDataInterface->retrieveHeadPos(raw_headpos);
                    mRawDataInterface->ExecuteHeadMode(2);
                    scanHead(raw_headpos.pitch,110);

                    // If scenario is finished, return to base position
                    if(elevator_scenario_state >= END){
                        mRawDataInterface->ExecuteHeadMode(0);
                        mRawDataInterface->ExecuteHeadPos(0,0,1);
                    }
                }
                else{
                    // If turning speed is 0, go to position mode and put head at normal angle to see in front
                    mRawDataInterface->ExecuteHeadMode(0);
                    mRawDataInterface->ExecuteHeadPos(0,0,1);
                }
            } else{
                // If turning speed is not 0, put head in speed mode (to control speed) and set its
                // speed to turning speed + add a difference in angle to anticipate and converge to
                // a certain angle : angle requested is rot_speed + Variable '-head_yaw' to regulate
                // speed/smoothness like a basic PD controller. No special need for PID at the moment.
                // Seems to be the best way to efficiently control the head
                mRawDataInterface->ExecuteHeadMode(2);
                mRawDataInterface->retrieveHeadPos(raw_headpos);
                mRawDataInterface->ExecuteHeadSpeed(w + (w-raw_headpos.yaw),0,1);
            }

            ALOGD("command safe: (%f,%f)", v, w);
        }

		float AlgoElevator::runTime()
		{
			std::lock_guard<std::mutex> lock(mMutexTimer);
			return m_ptime;
		}

		bool AlgoElevator::showScreen(void* pixels) // canvas 640x360, RGBA format
		{
			{
				std::lock_guard<std::mutex> lock(mMutexDisplay);
				if(mDisplayIm.empty())
					return false;
				cv::cvtColor(mDisplayIm, mDisplayData, CV_BGR2RGBA);
			}
			memcpy(pixels, (void *)mDisplayData.data, mDisplayData.cols * mDisplayData.rows * 4);
			return true;
		}

		void AlgoElevator::setDisplayData()
		{
			std::lock_guard<std::mutex> lock(mMutexDisplay);
			mDisplayIm = canvas.clone();
		}

        void AlgoElevator::createFolder(std::string new_folder_name)
        {
            std::string cmd_str_rm = "rm -rf \"" + new_folder_name + "\"";
            system(cmd_str_rm.c_str());  
            ALOGD("Command %s was executed. ", cmd_str_rm.c_str());          
            std::string cmd_str_mk = "mkdir \"" + new_folder_name + "\"";
            system(cmd_str_mk.c_str());
            ALOGD("Command %s was executed. ", cmd_str_mk.c_str());
        }


        bool AlgoElevator::initLocalMapping()
        {
            if(m_p_local_mapping) {
                delete m_p_local_mapping;
                m_p_local_mapping = NULL;
            }
            float mapsize = 6.0;
            float m_map_resolution = 0.05;
            m_p_local_mapping = new ninebot_algo::local_mapping::LocalMapping(mapsize, m_map_resolution);
            if(m_p_local_mapping == NULL)
                return false;
            CalibrationInfoDS4T calib;
            mRawDataInterface->getCalibrationDS4T(calib);
            float fx = calib.depth.focalLengthX;
            float fy = calib.depth.focalLengthY;
            float px = calib.depth.pricipalPointX;
            float py = calib.depth.pricipalPointY;
            m_p_local_mapping->setLidarRange(5.0);
            m_p_local_mapping->setLidarMapParams(0.6, true);
            m_p_local_mapping->setDepthCameraParams(px, py, fx, fy, 1000);
            m_p_local_mapping->setDepthRange(3.5, 0.35, 0.9, 0.1);
            m_p_local_mapping->setDepthMapParams(1.0, 10, false, -1);
            m_p_local_mapping->setUltrasonicRange(0.5);
            m_map_width =  mapsize / m_map_resolution;
            m_map_height = mapsize / m_map_resolution;

            return true;
        }

        bool AlgoElevator::prepare_localmap_and_pose_for_controller_g1() {
            // generate mapping
            mRawDataInterface->retrieveDepth(raw_depth, true);

            if(raw_depth.timestampSys == 0) {
                ALOGD("localmap: depth wrong");
                return false;
            }
            StampedMat local_depth(raw_depth.image, raw_depth.timestampSys);

            // merge all TF requests
            ninebot_tf::vector_req_t reqList;
            ninebot_tf::vector_tf_msg_t resList;
            reqList.push_back(ninebot_tf::tf_request_message("base_center_ground_frame", "world_odom_frame", raw_depth.timestampSys, 500));
            reqList.push_back(ninebot_tf::tf_request_message("rsdepth_center_neck_fix_frame", "world_odom_frame", raw_depth.timestampSys, 500));
            mRawDataInterface->getMassiveTfData(&reqList,&resList);
            if(resList.size()!=2){
                ALOGE("getMassiveTfData wrong");
                return false;
            }

            // clear history 
            // m_p_local_mapping->clearMap();

            ninebot_tf::tf_message tf_msg = resList[0];
            ninebot_tf::tf_message tf_msg2 = resList[1];
            if(tf_msg.err==0 && tf_msg2.err==0){
                StampedPose odom_pos = tfmsgTo2DPose(tf_msg);
                Eigen::Isometry3f depth_pose = PoseToCamcoor(tfmsgToPose(tf_msg2));
                // process depth
                m_p_local_mapping->processDepthFrame(local_depth, odom_pos, depth_pose);
            }
            else{
                ALOGE("localmap: Could not find tf pose with specified depth ts, error code: %d, %d", tf_msg.err, tf_msg2.err);
            }

            // StampedFloat ultrasonic;
            mRawDataInterface->retrieveUltrasonic(raw_ultrasonic);
            m_ultrasonic_buffer.push_back(raw_ultrasonic.value);
            if (m_ultrasonic_buffer.size() > 3) {
                m_ultrasonic_buffer.pop_front();
            }
            float ultrasonic_sum = 0;
            for (auto ultrasonic_element : m_ultrasonic_buffer) {
                ultrasonic_sum += ultrasonic_element;
            }
            m_ultrasonic_average = ultrasonic_sum / m_ultrasonic_buffer.size(); 

            // Uses depth camera + ultrasonic sensor for mapping
            m_p_local_mapping->getDepthMapWithFrontUltrasonic(m_local_map, false, tfmsgTo2DPose(resList[0]), m_ultrasonic_average);

            return true;
        }

        bool AlgoElevator::detectPersonFromMap() {
            m_persons_map = m_local_map.clone();
            int erosion_size = 1;
            cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                cv::Point(erosion_size, erosion_size));
            cv::dilate(m_persons_map, m_persons_map, element);
            cv::dilate(m_persons_map, m_persons_map, element);
            cv::erode(m_persons_map, m_persons_map, element);
            cv::dilate(m_persons_map, m_persons_map, element);
            cv::dilate(m_persons_map, m_persons_map, element);
            cv::erode(m_persons_map, m_persons_map, element);
            cv::erode(m_persons_map, m_persons_map, element);

            cv::Mat labels;
            cv::Mat stats;
            cv::Mat centroids;
            cv::connectedComponentsWithStats(m_persons_map, labels, stats, centroids, 8, CV_16U);

            m_persons.clear();
            for (int i = 0; i < stats.rows; i++)
            {
                int x = stats.at<int>(cv::Point(0, i));
                int y = stats.at<int>(cv::Point(1, i));
                int w = stats.at<int>(cv::Point(2, i));
                int h = stats.at<int>(cv::Point(3, i));
                int a = stats.at<int>(cv::Point(4, i));
                if (a < 40 && w < 6 && h < 6) {
                    cv::Scalar color(0, 0, 0);
                    cv::rectangle(m_persons_map, cv::Point2f(x, y), cv::Point2f(x+w, y+h), color);
                }
                else if (a > 1000) {
                    
                }
                else {
                    std::pair<int,int> position;
                    if (findFront(m_local_map,std::make_pair(x + std::round(w/2),y + std::round(h/2)),position,3)){
                        m_persons.push_back(position);
                    }
                }
            }

            return true;
        }

        // Display the field of view on an image
        void AlgoElevator::addShowFov(float cur_angle, float fov_angle, cv::Mat &show_img, float radius_per_meter)
        {
            cv::Point2f center = cv::Point2f(show_img.cols / 2, show_img.rows / 2);
            cv::Point2f left, right;
            float range = 50;
            if (cur_angle > PI)
                cur_angle -= 2 * PI;
            if (cur_angle < -PI)
                cur_angle += 2 * PI;
            left.x = center.x + range * cos(cur_angle - fov_angle / 2);
            left.y = center.y + range * sin(cur_angle - fov_angle / 2);

            right.x = center.x + range * cos(cur_angle + fov_angle / 2);
            right.y = center.y + range * sin(cur_angle + fov_angle / 2);

            cv::line(show_img, center, left, cv::Scalar(0, 0, 255), 1, cv::LINE_4);
            cv::line(show_img, center, right, cv::Scalar(0, 0, 255), 1, cv::LINE_4);

            if (radius_per_meter > 0) {
                cv::circle(show_img, center, 1 * radius_per_meter, cv::Scalar(255, 160, 160), 1.5);
                cv::circle(show_img, center, 2 * radius_per_meter, cv::Scalar(255, 160, 160), 1.5);
                cv::circle(show_img, center, 3 * radius_per_meter, cv::Scalar(255, 160, 160), 1.5);
                cv::line(show_img, cv::Point2f(show_img.cols / 2, 0), cv::Point2f(show_img.cols / 2, show_img.rows), cv::Scalar(255, 160, 160), 1.5);
                cv::line(show_img, cv::Point2f(0, show_img.rows / 2), cv::Point2f(show_img.cols, show_img.rows / 2), cv::Scalar(255, 160, 160), 1.5);
            }

            cv::Point2f loomo_left_front, loomo_right_front, loomo_left_rear, loomo_right_rear;
            loomo_left_front.x = center.x + 0.2*cos(cur_angle)*radius_per_meter - 0.3*sin(cur_angle)*radius_per_meter;
            loomo_left_front.y = center.y + 0.2*sin(cur_angle)*radius_per_meter + 0.3*cos(cur_angle)*radius_per_meter;
            loomo_right_front.x = center.x + 0.2*cos(cur_angle)*radius_per_meter + 0.3*sin(cur_angle)*radius_per_meter;
            loomo_right_front.y = center.y + 0.2*sin(cur_angle)*radius_per_meter - 0.3*cos(cur_angle)*radius_per_meter;
            loomo_left_rear.x = center.x - 0.7*cos(cur_angle)*radius_per_meter - 0.2*sin(cur_angle)*radius_per_meter;
            loomo_left_rear.y = center.y - 0.7*sin(cur_angle)*radius_per_meter + 0.2*cos(cur_angle)*radius_per_meter;
            loomo_right_rear.x = center.x - 0.7*cos(cur_angle)*radius_per_meter + 0.2*sin(cur_angle)*radius_per_meter;
            loomo_right_rear.y = center.y - 0.7*sin(cur_angle)*radius_per_meter - 0.2*cos(cur_angle)*radius_per_meter;

            cv::line(show_img, loomo_left_front, loomo_right_front, cv::Scalar(100, 200, 0), 2, cv::LINE_4);
            cv::line(show_img, loomo_left_front, loomo_left_rear, cv::Scalar(100, 200, 0), 2, cv::LINE_4);
            cv::line(show_img, loomo_right_rear, loomo_right_front, cv::Scalar(100, 200, 0), 2, cv::LINE_4);
            cv::line(show_img, loomo_left_rear, loomo_right_rear, cv::Scalar(100, 200, 0), 2, cv::LINE_4);
            cv::circle(show_img, loomo_right_front, 1, cv::Scalar(0, 0, 255), 3);
        }

        cv::Mat AlgoElevator::mapToShow(const cv::Mat & map) {
            cv::Mat show = cv::Mat::zeros(map.rows, map.cols, CV_8UC3);

            cv::Mat fix_map;
            int height, width, sel;
            Eigen::Vector3d mapCenter;

            // 2 lines can be used for displaying the bigger map
            m_p_local_mapping->getFixMap( fix_map, mapCenter, width, height, sel=0);
            cv::Mat show2 = cv::Mat::zeros(fix_map.rows, fix_map.cols, CV_8UC3);

            // Position of start, elevator door, elevator base goal, and calculated elevator goal (using obstacles)
            // Calculated for on-screen display
            // Positions can move based on odometry error as there is no map of the area
            double start_x;
            double start_y;
            double door_x;
            double door_y;
            double elevator_x;
            double elevator_y;
            double best_elevator_x;
            double best_elevator_y;

            // If we have started movement (and determined the goals), calculate position of goals
            if (elevator_scenario_state > 0) {
                StampedTwist odom;
                mRawDataInterface->retrieveOdometry(odom,-1);
                Eigen::Vector3f current_pos(odom.twist.pose.x, odom.twist.pose.y, odom.twist.pose.orientation);

                // Get the distance in meters between the robot and the 4 positions to draw on map
                start_x = start_position[0]-current_pos[0];
                start_y = start_position[1]-current_pos[1];
                door_x = door_position[0]-current_pos[0];
                door_y = door_position[1]-current_pos[1];
                elevator_x = elevator_position[0]-current_pos[0];
                elevator_y = elevator_position[1]-current_pos[1];
                best_elevator_x = best_elevator_position[0]-current_pos[0];
                best_elevator_y = best_elevator_position[1]-current_pos[1];
            }

            // For each pixel, determine if it is an obstacle or not, & if it is a goal & assign
            // the appropriate colour
            for (int i = 0; i < map.rows; i++)
            {
                for (int j = 0; j < map.cols; j++)
                {
                    // Draws obstacle pixels
                    uchar map_value = map.at<uchar>(i, j);
                    if (map_value == 255)
                    {
                        show.at<cv::Vec3b>(i, j) = cv::Vec3b(127, 255, 127);
                    }
                    else
                    {
                        show.at<cv::Vec3b>(i, j) = cv::Vec3b(255 - 2.5 * map_value, 255 - 2.5 * map_value, 255 - 2.5 * map_value);
                    }

                    if(elevator_scenario_state>0){
                        // If starting position is in current pixel area, colour pixel blue
                        // Uses conversions to move from distance coordinates to map coordinates
                        // Draws a 5x5 pixel square around the goal
                        if((start_y*map.rows/6) > -map.rows/2 + i-3 && (start_y*map.rows/6) < -map.rows/2 + i+3){
                            if ((start_x*map.cols/6) > -map.cols/2 + j-3 && (start_x*map.cols/6) < -map.cols/2 + j+3){
                                show.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
                            }
                        }
                        // Colour door position (elevator door) on screen in red
                        if((door_y*map.rows/6) > -map.rows/2 + i-3 && (door_y*map.rows/6) < -map.rows/2 + i+3){
                            if ((door_x*map.cols/6) > -map.cols/2 + j-3 && (door_x*map.cols/6) < -map.cols/2 + j+3){
                                show.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
                            }
                        }
                        // Colour the basic elevator position pixels in green
                        if((elevator_y*map.rows/6) > -map.rows/2 + i-3 && (elevator_y*map.rows/6) < -map.rows/2 + i+3){
                            if ((elevator_x*map.cols/6) > -map.cols/2 + j-3 && (elevator_x*map.cols/6) < -map.cols/2 + j+3){
                                show.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 255, 0);
                            }
                        }
                        // Colour the desired elevator final goal in purple
                        if((best_elevator_y*map.rows/6) > -map.rows/2 + i-3 && (best_elevator_y*map.rows/6) < -map.rows/2 + i+3){
                            if ((best_elevator_x*map.cols/6) > -map.cols/2 + j-3 && (best_elevator_x*map.cols/6) < -map.cols/2 + j+3){
                                show.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 255);
                            }
                        }
                    }
                }
            }

            return show.clone();
        }

        bool AlgoElevator::findFront(const cv::Mat & map, const std::pair<int,int> & proposal, std::pair<int,int> & result, int region) {
            float min_distance = 10000000;
            float row_center = (map.rows-1)/2;
            float col_center = (map.cols-1)/2;
            for (int row = std::max(proposal.first - region,0); row != std::min(proposal.first + region,map.rows); row++) {
                for (int col = std::max(proposal.second - region,0); col != std::min(proposal.second + region,map.cols); col++) {
                    uchar prob = map.at<uchar>(col,row);
                    if(prob > 0) {
                        float distance = std::sqrt(std::pow((row - row_center),2) + std::pow((col - col_center),2));
                        if (distance < min_distance){
                            result.first = row;
                            result.second = col;
                            min_distance = distance;
                        }
                    }
                }
            }
            if (min_distance < 10000000) {
                return true;
            }
            else {
                return false;
            }
        }

        // Turns the head of the robot. Must make sure that the head is in the correct mode before
        // calling this function (speed/position)
        void AlgoElevator::scanHead(float pitch, float yaw_limit_degree)
        {
            // rotate head
            StampedHeadPos raw_headpos;
            mRawDataInterface->retrieveHeadPos(raw_headpos);

            float angle_target = abs(yaw_limit_degree)*3.1415/180;

            if(scan_round==0){
                if(raw_headpos.yaw>-angle_target+0.25)
                    //mRawDataInterface->ExecuteHeadPos(-angle_target,pitch,0);

                    // Control scan head function with speed for more control
                    // Speed decreases from approximately 1rad/s to 0.5rad/s linearly before changing direction
                    mRawDataInterface->ExecuteHeadSpeed(-1+(-angle_target+raw_headpos.yaw)/-4/angle_target,0,1);
                else
                    scan_round = 1;
            }
            else{
                if(raw_headpos.yaw<angle_target-0.25)
                    //mRawDataInterface->ExecuteHeadPos(angle_target,pitch,0);
                    mRawDataInterface->ExecuteHeadSpeed(1+(angle_target+raw_headpos.yaw)/-4/angle_target,0,1);
                else
                    scan_round = 0;
            }
        }

        bool AlgoElevator::initLocalPlanner()
        {
            if (m_p_model_based_planner) {
                delete m_p_model_based_planner;
                m_p_model_based_planner = NULL;
            }
            vector<Eigen::Vector2f> ref_trajectory;
            ref_trajectory.clear();
            m_p_model_based_planner = new ninebot_algo::model_based_planner::ModelBasedPlanner(ref_trajectory,true,TypeRobot::G1);

            m_p_model_based_planner->set_min_dist_to_obs(0.02);
            m_p_model_based_planner->set_max_dist_offset(0.3);
            m_p_model_based_planner->set_min_linear_velocity(-1);

            ALOGD("initLocalPlanner");  
            return true;
        }

        // Initialisation for elevator scenario
        void AlgoElevator::startPathElevator(){

		    // Set state to 1 == go towards elevator door
            elevator_scenario_state = MOVE_TO_DOOR;

            // Update current scenario based on current state
            elevatorScenarioUpdate();

            // Allows to enter step function
            m_has_path_init = true;
        }


        // Function that changes the goal of the robot based on its state (in order)
        // Contains necessary changes for each step of the process from going to the elevator to exiting
        void AlgoElevator::elevatorScenarioUpdate(){


            std::vector<Eigen::Vector2f> path;


            // 1) create elevator and door locations and move to door
            // If path is blocked, should wait at a distance if we can see it already
            // Must start robot at a certain position and already mark distances because no map is available
            // Must remember original position, using odometry, to find its way back (out of elevator)

            if(elevator_scenario_state == MOVE_TO_DOOR){
                // set path
                // get current pos and remember
                StampedTwist odom;
                mRawDataInterface->retrieveOdometry(odom,-1);
                Eigen::Vector3f pos_start(odom.twist.pose.x, odom.twist.pose.y, odom.twist.pose.orientation);
                start_position = pos_start;
                elevator_position = pos_start;

                // create path using orientation (x = forward (in meters), y = positive = left)
                // Distance with respect to robot state (position + orientation)
                double dist_x = 2;
                double dist_y = -1;

                double dist_elev_x = 2;
                double dist_elev_y = -3;

                // Clear path and create new path of 2 points in straight line towards elevator door
                // It is possible to deviate slightly from the points
                Eigen::Vector2f temp;
                path.clear();
                for (float d = 0; d <= 1; d+=0.5) {
                    temp << d*dist_x * cos(pos_start[2]) + d*dist_y * sin(-pos_start[2]) + pos_start(0), d*dist_x * sin(pos_start(2)) + d*dist_y * cos(pos_start[2]) + pos_start(1);
                    path.push_back(temp);
                }

                // Save position of the door as global variable
                Eigen::Vector3f pos_door(dist_x * cos(pos_start[2]) + dist_y * sin(-pos_start[2]) + pos_start[0], dist_x * sin(pos_start[2]) + dist_y * cos(pos_start[2]) + pos_start[1], 0);
                door_position = pos_door;

                // Save position of the elevator as global variable
                Eigen::Vector3f pos_elevator(dist_elev_x * cos(pos_start[2]) + dist_elev_y * sin(-pos_start[2]) + pos_start[0], dist_elev_x * sin(pos_start[2]) + dist_elev_y * cos(pos_start[2]) + pos_start[1], 0);
                elevator_position = pos_elevator;

                // Update the path
                m_p_model_based_planner->reset_global_reference(path);
            }

            // 2 door reached, move in the elevator to preferred location by searching for best option
            // Set the current goal which is already known in advance
            // Make a search in the area of goal, looking at obstacles to find best position and go
            // Continue searching for spots with time

            if (elevator_scenario_state == MOVE_IN_ELEVATOR){
                setNewElevatorGoal();
                ALOGD("New elevator goal set");

                // during update, check for best position until goal reached
                // if not computationally too expensive
                // Need to update the goal as the environment changes
            }

            // 3 goal reached, turn around
            // possibly set the goal to a few cm behind robot to turn around
            // if elevator empty, go to door and wait

            if(elevator_scenario_state == TURN_TOWARDS_DOOR){
                // Turn to face the elevator door

                // Turning is managed in the step function for the elevator

                path.clear();

                //m_p_model_based_planner->reset_global_reference(path);
            }

            if(elevator_scenario_state == WAIT){

                // Optional function to make the robot wait. For simulating time while door closes for example.

                this->safeControl(0,0);

                //usleep(4000000);
            }

            // 5 door opens
            // go back to the original position (without looking at orientation)

            if(elevator_scenario_state == GO_TO_STARTING_POSITION){

                Eigen::Vector2f temp;
                path.clear();

                // Give single point to the path (should have good knowledge of the obstacles
                // since it already passed by)
                temp << start_position(0), start_position(1);
                path.push_back(temp);

                m_p_model_based_planner->reset_global_reference(path);
            }

            // Robot returned to starting point ---> stop
            if(elevator_scenario_state == END){
                ALOGD("Enter state 6: arrived: END");
                this->safeControl(0,0);
                elevator_scenario_state = 7; // State with no commands
                m_has_path_init = false;
            }

		}

        // Step function of local planner that gives robot its speed, depending on where it wants to go
        void AlgoElevator::stepLocalPlanner()
        {
            Eigen::Vector3f cur_pose(raw_odometry.twist.pose.x, raw_odometry.twist.pose.y, raw_odometry.twist.pose.orientation);
            Eigen::Vector2f cur_vel(raw_velocity.vel.linear_velocity, raw_velocity.vel.angular_velocity);

            ALOGD("stepLocalPlanner: cur_pose: %.2f,%.2f,%.2f", cur_pose[0],cur_pose[1],cur_pose[2]);
            ALOGD("stepLocalPlanner: cur_vel: %.2f,%.2f", cur_vel[0],cur_vel[1]);
            ALOGD("stepLocalPlanner: m_local_map row, col, type: %d,%d,%d", m_local_map.rows, m_local_map.cols, m_local_map.type());

            m_planner_status = m_p_model_based_planner->local_planner_process(cur_pose, cur_vel, m_local_map);


            // If state == going towards elevator door or in elevator or back to starting position
            if (elevator_scenario_state == MOVE_TO_DOOR || elevator_scenario_state == MOVE_IN_ELEVATOR
                || elevator_scenario_state == GO_TO_STARTING_POSITION) {

                Eigen::Vector2f vel_planned(0.0f, 0.0f);
                if (m_planner_status == model_based_planner::planner_normal) {

                    // Normal planner ---> retrieve its velocity output
                    vel_planned = m_p_model_based_planner->get_planner_velocity();
                    if (elevator_scenario_state == MOVE_IN_ELEVATOR) {
                        setNewElevatorGoal();
                        ALOGD("Normal new elevator goal");
                    }
                } else if (m_planner_status == model_based_planner::goal_have_reached) {
                    // Goal reached ---> go to next phase
                    elevator_scenario_state += 1;
                    elevatorScenarioUpdate();
                } else {
                    ALOGD("stepLocalPlanner: abnormal status = %d", m_planner_status);

                    // If abnormal status while moving in elevator, create new elevator goal
                    // Reasoning is that it means that the goal is unreachable and must be changed
                    if (elevator_scenario_state == MOVE_IN_ELEVATOR) {
                        setNewElevatorGoal();
                        ALOGD("Another new elevator goal");
                    }
                }
                ALOGD("stepLocalPlanner: vel_planned: %.2f,%.2f", vel_planned[0], vel_planned[1]);

                this->safeControl(vel_planned[0], vel_planned[1]);
            }

            // Turn around to face elevator door phase
            else if(elevator_scenario_state == TURN_TOWARDS_DOOR){

                double target_angle;

                StampedTwist odom;
                mRawDataInterface->retrieveOdometry(odom,-1);
                Eigen::Vector3f current_pos(odom.twist.pose.x, odom.twist.pose.y, odom.twist.pose.orientation);

                // distance to travel to go to door, used to know the desired orientation
                double dist_x = door_position[0]-current_pos[0];
                double dist_y = door_position[1]-current_pos[1];

                // Take care of exceptions (division by 0)
                if (dist_x == 0){
                    if(dist_y > 0) target_angle = 3.1415/2;
                    else target_angle = -3.1415/2;
                }
                else target_angle = atan(dist_y/dist_x);

                // Atan only works for half of trig circle, take care of 2nd half --->
                if (dist_x < 0) {
                    if (target_angle > 0){
                        target_angle -= 3.1415;
                    }
                    else target_angle += 3.1415;
                }

                // Obtain robot current orientation (retrieveOdometry() doesn't work)
                ninebot_tf::tf_message tf_msg = mRawDataInterface->GetTfBewteenFrames("base_center_ground_frame", "world_odom_frame", -1, 500);
                StampedPose odom_pos = tfmsgTo2DPose(tf_msg);
                float yaw = odom_pos.pose.orientation;


                // Choose how to turn based on current angle and desired angle
                if(yaw < 0){
                    if(target_angle > yaw && target_angle < yaw + 3.1415) {
                        //turn left
                        this->safeControl(0,0.4);
                    }
                    else {
                        // turn right
                        this->safeControl(0,-0.4);
                    }
                    // if within range, set speed to 0, go to next phase
                    if(yaw > target_angle - 3.1415/10 && yaw < target_angle + 3.1415/10){
                        this->safeControl(0,0);
                        elevator_scenario_state += 1;
                        elevatorScenarioUpdate();
                    }
                }
                else {
                    if (target_angle < yaw && target_angle > yaw - 3.1415){
                        // turn right
                        this->safeControl(0,-0.4);
                    }
                    else {
                        // turn left
                        this->safeControl(0,0.4);
                    }
                    // if within range, set speed to 0, go to next phase
                    if(yaw > target_angle - 3.1415/10 && yaw < target_angle + 3.1415/10){
                        this->safeControl(0,0);
                        elevator_scenario_state += 1;
                        elevatorScenarioUpdate();
                    }
                }
            }
            // If path state between turning to face elevator and moving to elevator
            // Sometimes used as a pause (waiting phase) but not essential
            else if (elevator_scenario_state == WAIT) {
                // Here: finished waiting
                elevator_scenario_state += 1;
                elevatorScenarioUpdate();
            }

        }


        // return index of best goal, using the local map, current goal and width of interest for searching
        cv::Point AlgoElevator::getBestGoal(cv::Mat localMap, int goal_x, int goal_y, int search_width){

		    // Matrix that says whether there is obstacle or not
		    cv::Mat labelMap(localMap.rows, localMap.cols, CV_8UC1);

		    // Matrix containing distance to obstacles: largest value is the point with most space
		    cv::Mat outputMat(localMap.rows, localMap.cols, CV_8UC1);
            cv::Mat maskMat(localMap.rows, localMap.cols, CV_8UC1);


            // Create the obstacle matrix
            for (int i = 0; i < localMap.rows; ++i) {
                for (int j = 0; j < localMap.cols; ++j) {
                    if(localMap.at<uchar>(i,j) > 20){
                        labelMap.at<uchar>(i,j)= 0;
                    } else labelMap.at<uchar>(i,j) = 1;
                }
            }

            // Creates matrix of distance of each pixel from nearest obstacle pixel (L2 distance)
            // using the obstacle matrix. Distance is in pixels
            cv::distanceTransform(labelMap, outputMat, cv::DIST_L2, cv::DIST_MASK_5, CV_32F);

            // If current goal is at least x/2 meters from obstacles, keep the current goal and return location
            // Thought process is that we are far away from obstacles (not-human aware, just looks for space)
            if (outputMat.at<uchar>(goal_y, goal_x) > 0.8*m_local_map.rows/6){
                cv::Point location;
                location.x = goal_x;
                location.y = goal_y;
                return location;
            }

            // Min and max locations (distance from obstacles)
            double min, max;
            cv::Point minLoc;
            cv::Point maxLoc;

            // Creates a matrix with value 1 in a square around interest point using the width
            // This serves as a mask to filter out points too far from elevator
            // The square orientation depends on elevator orientation in the local map but in the future,
            // should search the whole elevator if the size is given (not human aware decision making here)
            for (int i = 0; i < labelMap.rows; ++i) {
                for (int j = 0; j < labelMap.cols; ++j) {
                    if (i > goal_x - search_width/2 && i< goal_x + search_width/2 &&
                            j > goal_y - search_width/2 && j< goal_y + search_width/2){
                        maskMat.at<uchar>(j,i) = 1;
                    }
                    else maskMat.at<uchar>(j,i) = 0;
                }
            }

            // Looks for location of max distance from obstacles in the correct search area defined by mask
            cv::minMaxLoc(outputMat, &min, &max, &minLoc, &maxLoc, maskMat);

            // Save the best location as a point and return
            cv::Point location;
            location.x = maxLoc.x;
            location.y = maxLoc.y;

            return location;
		}

		// Function creates new goal location in elevator based on closeness to objects
        // Finds best location within a given area
		void AlgoElevator::setNewElevatorGoal() {

            std::vector<Eigen::Vector2f> path;

            StampedTwist odom;
            mRawDataInterface->retrieveOdometry(odom,-1);
            Eigen::Vector3f current_pos(odom.twist.pose.x, odom.twist.pose.y, odom.twist.pose.orientation);

            // Calculate distance towards main elevator objective
            double elevator_x = elevator_position[0]-current_pos[0];
            double elevator_y = elevator_position[1]-current_pos[1];


            // Width of the square around main elevator objective to search for potential best local objective
            int width = 2 * m_local_map.rows/6; // 2* = 2 meters (1 each way)

            // Locate the best point for placement and return it
            // Elevator distance transformed to local map coordinates
            cv::Point new_goal = getBestGoal(m_local_map, (int) (m_local_map.cols/2 + elevator_x*m_local_map.cols/6), (int) (m_local_map.rows/2 + elevator_y*m_local_map.rows/6), width);

            // Transform goal location on local map to distance from robot
            elevator_x = (double) (new_goal.x-m_local_map.cols/2)*6/m_local_map.cols;
            elevator_y = (double) (new_goal.y-m_local_map.rows/2)*6/m_local_map.rows;


            // Create robot path (straight line of 10 evenly spaced points to elevator goal)
            // Robot can deviate from straight line within a set limit
            Eigen::Vector2f temp;
            path.clear();

            temp << elevator_x  + current_pos(0), elevator_y + current_pos(1);
            path.push_back(temp);

            // Update global variable of desired elevator position. Used for on screen display.
            Eigen::Vector3f pos_elevator(elevator_x  + current_pos[0], elevator_y  + current_pos[1], 0);
            best_elevator_position = pos_elevator;

            // Give path to the robot
            m_p_model_based_planner->reset_global_reference(path);
		}

	} // namespace elevator_simulation_algo
} // namespace ninebot_algo
