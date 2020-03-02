#ifndef NINEBOT_ALGO_ELEVATOR_SIM_H
#define NINEBOT_ALGO_ELEVATOR_SIM_H

#include <fstream>
#include "AlgoBase.h"
#include "LocalMapping.h"
#include "model_based_planner.h"

#define MOVE_TO_DOOR 1
#define MOVE_IN_ELEVATOR 2
#define TURN_TOWARDS_DOOR 3
#define WAIT 4
#define GO_TO_STARTING_POSITION 5
#define END 6

namespace ninebot_algo
{
	/*! namespace of this algorithm  */
	namespace elevator_simulation_algo
	{

		/*! Define the configurable parameters for this algorithm  */
		typedef struct test_params
		{
			//char* test_config_filename;
			// ........................................

		} test_params_t;

		/*! class of this algorithm, derived from base class AlgoBase */
		class AlgoElevator : public AlgoBase {
		public:
            /*! Constructor of AlgoSocket
                 * @param rawInterface The pointer to the RawData object instantiated in main()
                 * @param run_sleep_ms The sleep time in millisecond for AlgoBase::run() - AlgoBase::step() is executed following a sleep in run()
            * @param isRender The bool switch to select whether or not do rendering work in step()
        	*/
			AlgoElevator(RawData* rawInterface, int run_sleep_ms, bool isRender=true):
				AlgoBase(rawInterface,run_sleep_ms,true){
				m_is_init_succed = false;
				m_ptime = 10;
				m_isRender = isRender;
                pose_isRecording = false;
				canvas = cv::Mat::zeros( cv::Size(640, 360), CV_8UC3 );
				m_is_stop = false;
				m_slide_event = 0;
				step_count = 0;
				motion_test = 0;
				motion_sign = 1;
				scan_round = 0;
				m_safety_control = true;
                m_p_local_mapping = NULL;
                m_p_model_based_planner = NULL;
                m_has_path_init = false;
			}
			~AlgoElevator();


			/*! Implement a complete flow of this algorithm, takes all inputs, run once, and output */
			virtual bool step();

			/*! Initialize current algorithm with configurable parameters */
			bool init(test_params_t cParams);
			/*! Return the runtime of step() in milliseconds, shall implement the calculation of this number in step(). This function is called by main(). */
			float runTime();
			/*! Copy the internal drawing content to the external display buffer. This function is called by main(), the internal canvas is maintained and updated in step() according to user's demands */
			bool showScreen(void* pixels);

            void startPoseRecord(std::string save_folder, int64_t save_time);
            void stopPoseRecord();

            void onEmergencyStop(bool is_stop);
            void onWheelSide(int event);

			// imu callback
			virtual void UpdateAccel(MTPoint val);
			virtual void UpdateGyro(MTPoint val);
			/*! enable sync test*/
			void toggleImgStream();
			void startMotionTest();
            // VLS test
            void setVLSopen(bool en);
            
            // online config
			void switchSafetyControl();

			void startPathElevator();

		private:
			/*! Copy internal canvas to intermediate buffer mDisplayIm */
			void setDisplayData();

            int step_count;
			int motion_test;
			int motion_sign;
            bool m_is_stop;
            int m_slide_event;

            bool pose_isRecording;
            int nStep;
			bool m_is_init_succed;
			// class variables
			StampedMat raw_fisheye, raw_depth, raw_color, raw_colords4;
			StampedOrientation raw_orientation;
			StampedIr raw_ir;
			StampedHeadPos raw_headpos;
			StampedHeadPos raw_maincampos;
			StampedBasePos raw_basepos;
			StampedFloat raw_ultrasonic;
			StampedTwist raw_odometry;
			StampedVelocity raw_velocity;
			StampedBaseWheelInfo raw_wheel_info;
			RawCameraIntrinsics raw_camerapara;
			CalibrationInfoDS4T calib;
			MTPoint raw_gyro;
			MTPoint raw_accel;
			int64_t t_old;
			bool m_isRender;
			std::mutex mMutexDisplay;
			cv::Mat canvas, mDisplayIm, mDisplayData;
			std::mutex mMutexTimer;
			float m_ptime;
			bool imgstream_en;
			int64_t m_timestamp_start;
			// algo parameters

			//void stepServer();
			std::string m_folder_socket;
			std::ofstream m_state_file;
        	void createFolder(std::string new_folder_name);

        	// local map
        	bool initLocalMapping();
        	bool prepare_localmap_and_pose_for_controller_g1();
			RawData* m_p_localmapping_rawdata;
            int m_map_width, m_map_height;
            ninebot_algo::local_mapping::LocalMapping *m_p_local_mapping;
            cv::Mat m_local_map, m_persons_map;

            // multi-person detection
            std::vector<std::pair<float, float>> m_persons;
            bool detectPersonFromMap();

            // visualization
            void addShowFov(float cur_angle, float fov_angle, cv::Mat &show_img, float radius_per_meter);
			cv::Mat mapToShow(const cv::Mat & map);
			bool findFront(const cv::Mat & map, const std::pair<int,int> & proposal, std::pair<int,int> & result, int region = 3);
		
            // head scan		
			void scanHead(float pitch, float yaw_limit_degree);
			int scan_round;

			// safety verification
			void safeControl(float v, float w);
			bool m_safety_control; 
			std::deque<float> m_ultrasonic_buffer;
			float m_ultrasonic_average;

			// obstacle avoidance 
			ninebot_algo::model_based_planner::ModelBasedPlanner *m_p_model_based_planner;
			ninebot_algo::model_based_planner::PLANNER_STATUS m_planner_status;
			bool m_has_path_init;
            bool initLocalPlanner();
            void stepLocalPlanner();

            // Elevator task
            void elevatorScenarioUpdate();
			cv::Point getBestGoal(cv::Mat localMap, int goal_x, int goal_y, int search_width);
			void setNewElevatorGoal();
            int elevator_scenario_state;
            Eigen::Vector3f start_position;
			Eigen::Vector3f door_position;
			Eigen::Vector3f elevator_position;
			Eigen::Vector3f best_elevator_position;
		};

	} // namespace elevator_simulation_algo
} // namespace ninebot_algo

#endif