#ifndef _ROBOT_GEOMETRY_H_
#define _ROBOT_GEOMETRY_H_

#include <Eigen/Core>
#include <mutex>

namespace ninebot_algo {
	
	enum TypeRobot
	{
		Cylinder = -1,
		G1 = 0,
		Go = 1,
		GX = 2
	};

	class RobotGeometry {

	public:

		RobotGeometry(TypeRobot type, float resolution, float clearance = 0.03);
		~RobotGeometry();

		bool set_robot_geometry(float clearance);

		Eigen::MatrixX2f get_robot_polygon();
		Eigen::MatrixX2f get_robot_interior();
		Eigen::MatrixX2f get_robot_front();
		Eigen::MatrixX2f get_robot_back();
		Eigen::MatrixX2f get_robot_left();
		Eigen::MatrixX2f get_robot_right();
		TypeRobot get_robot_type();

	private:
		TypeRobot m_type;
		float m_resolution;
		std::mutex m_mutex_io;

		Eigen::MatrixX2f m_polygon;
		Eigen::MatrixX2f m_interior;
		Eigen::MatrixX2f m_front;
		Eigen::MatrixX2f m_back;
		Eigen::MatrixX2f m_left;
		Eigen::MatrixX2f m_right;

		void set_robot_geometry_G1(float clearance);
		void set_robot_geometry_Go(float clearance);
		void set_robot_geometry_GX(float clearance);
		void set_robot_geometry_Cylinder(float clearance);

	};

}
#endif
