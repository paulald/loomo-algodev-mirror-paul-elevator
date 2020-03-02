#ifndef _MODEL_BASED_PLANNER_H_
#define _MODEL_BASED_PLANNER_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <string>
#include <mutex>
#include "robot_geometry.h"

//#define USE_VIN_PLANNER

#ifdef USE_VIN_PLANNER
class vin_planner;
#endif

namespace ninebot_algo {

	namespace local_planner
	{
		class trajectory_generator;
		class trajectory_evaluator;
		class trajectory;
		class local_planner_limits;
	}

	namespace model_based_planner {

		// sampling parameter
		#define M_P_NUM_V 7
		#define M_P_NUM_W 11

		enum PLANNER_STATUS
		{
			planner_unknown = 0, ///< brief unknown
			planner_normal = 1, ///< breif get normal trajectory
			goal_have_reached = 2, ///< breif reach the goal
			goal_not_find = 3,///< breif find no goal in search max range
			goal_invalid = 4, ///< breif all way points in obstacle
			update_failed = 5, ///< update state failed
			traj_invalid = 6
		};

		enum CELL_STATUS
		{
			free = 0,
			goal = 1,
			undetected = 2
		};

		//class __attribute__((visibility("default"))) ModelBasedPlanner
		class ModelBasedPlanner
		{
			public:
				ModelBasedPlanner(const std::vector<Eigen::Vector2f> &ref_trajectory, bool goal_gen_internal = true, TypeRobot type = TypeRobot::Go, float radius = 0.0f);
				~ModelBasedPlanner(); 

				// config settings
				void set_termination_orientation(float theta);
				void set_termination_alignment(bool is_aligned);
				void set_initialization_alignment(bool is_aligned);
				void set_max_linear_velocity(const float max_linear_velocity);
				void set_min_linear_velocity(const float min_linear_velocity);
				void set_max_angular_velocity(const float max_angular_velocity);
				void set_max_angular_acceleration(const float max_angular_velocity);
				void set_max_linear_acc(const float max_linear_acc);
				void set_reach_tolerance(float tolerance);
				void set_landing_tolerance(float tolerance);
				void set_landing_timeout(float max_trial);
				void set_min_dist_to_obs(float thres_dist);
				void set_max_orientation_offset(float thres_orientation);	// maximum orientation offset to pre-defined reference path
				void set_max_dist_offset(float thres_dist);					// maximum distance offset to pre-defined reference path
				void set_max_traj_product(float thres_product);
				void set_min_traj_length(float min_length);
				void set_max_traj_length(float max_length);
				void set_trajs_complexity(int level_complexity);
				void set_steps_tail_check(int num_steps);					// non-cylinder robot
				void set_thres_direct_rotation(float angle);
				void set_score_weight(const Eigen::VectorXf &score_weight);
				void set_sample_param(local_planner::local_planner_limits *limits, const Eigen::Vector2i& vsamples);

				// config file
				bool set_config(std::string filename);

				/*
				reset reference path 
				*/
				void reset_global_reference(const std::vector<Eigen::Vector2f>& ref_trajectory, bool goal_gen_internal = true);
				
				/*
				manipulate local goal point
				*/
				bool set_cur_goal(const std::pair<int, Eigen::Vector2f>& cur_goal);
				bool set_cur_goal_pose(const std::pair<int, Eigen::Vector3f>& cur_goal_pose);

				// active planning process
				PLANNER_STATUS local_planner_process(const Eigen::Vector3f &cur_pose, const Eigen::Vector2f &cur_vel, const cv::Mat &map, bool is_forward = true, bool is_active_plan = true);

				// active command verification process
				PLANNER_STATUS check_command_process(const Eigen::Vector3f &cur_pose, const Eigen::Vector2f &cur_vel, const cv::Mat &map, const float linear_velociy, const float angular_velocity, const float duration = 1.0);

				// get result
				std::vector<Eigen::Vector3f> get_planner_trajecotry();
				Eigen::Vector2f get_planner_velocity();
				float get_planner_cost();
				std::pair<int, Eigen::Vector2f> get_cur_goal();
				std::vector<local_planner::trajectory> get_pred_traj();
				std::vector<std::vector<Eigen::Vector3f>> get_sample_trajectories();
				cv::Mat get_blur_map();
				cv::Mat get_occu_map();

				// get status
				PLANNER_STATUS get_planner_status();

			private:
				void step_active_decision(const bool is_forward);
				void step_passive_decision(const bool is_forward);
				void step_fix_decision(const float linear_velocity, const float angular_velocity, const float duration);
				bool step_rotate_decision(float target_theta, float thres_residual);

				bool loadConfigFile(std::string filename);

				bool update_states(Eigen::Vector3f cur_pos, Eigen::Vector2f cur_vel, const cv::Mat &occupancy_map);
				void update_trajectories(const Eigen::Vector2f& goal_point, bool is_forward);

				// internal status 
				bool m_is_goal_gen_internal;
				bool m_is_init;
				bool m_is_initialization_aligned;

				// termination
				bool m_is_termination_aligned;
				float m_termination_orientation;

				// output 
				Eigen::Vector2f saturate_velocity(const Eigen::Vector2f & ori);
				float m_max_linear_velocity, m_max_angular_velocity, m_max_linear_acc, m_min_linear_velocity, m_max_angular_acc;

				// sub-module 
				local_planner::local_planner_limits *m_limits;
				local_planner::trajectory_generator *m_traj_gen;
				local_planner::trajectory_evaluator *m_traj_eval;
				local_planner::trajectory *m_best_traj;
				Eigen::Vector2i m_vsamples;

				// input members 
				std::vector<Eigen::Vector2f> m_ref_trajectory;
				Eigen::Vector3f m_cur_pos;
				Eigen::Vector2f m_cur_vel;
				cv::Mat m_occu_map;

				// intermediate member 
				Eigen::Vector2f m_next_goal;
				int m_cnt_rot_invalid;
				std::mutex m_mutex_io;

				// output member 
				std::vector<local_planner::trajectory> m_pred_trajs;
				PLANNER_STATUS m_planner_status;
				float m_cost_traj;

				// config member 
				float m_reach_tolerance;
				float m_landing_tolerance;
				float m_landing_timeout;
				int m_complexity_level;

				// parameters
				static constexpr float M_W_ACC_MAX = 0.3;
				static constexpr float M_TRAJ_V_MAX = 0.8;
				//static constexpr float M_TRAJ_V_MIN = 0.1;
                static constexpr float M_TRAJ_V_MIN = 0.0;
				static constexpr float M_TRAJ_W_MAX = 1.0;		// (M_P_NUM_W-1) / 2 * deg2rad(10)
				static constexpr float M_TRAJ_W_MIN = -1.0;
				static const int M_K_SZ_LOCAL_MAP = 120;
				static constexpr float M_REACH_THRESHOLD = 0.5;
				static constexpr float M_LANDING_THRESHOLD = 0.8;

				// planner configure
				enum PARAM_SETTING
				{
					MAX_LINEAR_VEL,
					MAX_LINEAR_ACC,
					MIN_DIST_TO_OBS,
					MAX_NAVI_OFFSET,
					REACH_TOLERANCE,
					LANDING_TOLERANCE,
					LANDING_TIMEOUT,
					MIN_TRAJ_LENGTH,
					MAX_TRAJ_PRODUCT,
					MAX_ORIENTATION_OFFSET,
					LEVEL_TRAJS_COMPLEXITY,
					STEPS_BORDER_CHECK,
					THRES_DIRECT_ROTATION,
					MAX_ANGULAR_VEL,
					MIN_LINEAR_VEL,
				};


#ifdef USE_VIN_PLANNER
			public:
				bool crop_reward_map_from_occu_map(cv::Mat &reward_map, const cv::Mat &occu_map, float theta, const Eigen::Vector2f cur_goal_point);
				bool check_policy_deviation(Eigen::Vector2f cur_goal_local, float cur_state_theta, float w_to_control);
			
			private:
				vin_planner *m_vin_planner;
				static constexpr float M_K_REWARD_GOAL = 1.1;
				static const int M_K_SZ_CROP = 28;
				static constexpr float M_K_PAYOUT = -0.01;
				static constexpr float M_K_CROP_LEFT = -0.4;
				static constexpr float M_K_CROP_RIGHT = 2.4;
				static constexpr float M_K_CROP_UP = -1.4;
				static constexpr float M_K_CROP_DOWN = 1.4;
				static constexpr float M_K_VIN_RESOLUTION = 0.1;
#endif

		};
	};
};
#endif
