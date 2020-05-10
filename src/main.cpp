#include <fstream>
// #include <iostream>
#include <math.h>
// #include <stdlib.h>


using namespace std;

struct Vector3
{
	Vector3() = default;

	Vector3(float xParam, float yParam, float zParam) :
		x(xParam),
		y(yParam),
		z(zParam)
	{
	}

	float length()
	{
		return sqrt(lengthSquared());
	}

	float lengthSquared()
	{
		return x * x + y * y + z * z;
	}

	void normalize()
	{
		float len = length();
		x /= len;
		y /= len;
		z /= len;
	}

	float x;
	float y;
	float z;
};

float dot(const Vector3 & lhs, const Vector3 & rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Vector3 cross(const Vector3 & lhs, const Vector3 & rhs)
{
	return Vector3(
		lhs.y * rhs.z - lhs.z * rhs.y,
		lhs.z * rhs.x - lhs.x * rhs.z,
		lhs.x * rhs.y - lhs.y * rhs.x
	);
}


Vector3 operator+(Vector3 v)
{
	return Vector3(v.x, v.y, v.z);
}

Vector3 operator-(Vector3 v)
{
	return Vector3(-v.x, -v.y, -v.z);
}

void operator+=(Vector3 & vLeft, Vector3 vRight)
{
	vLeft.x += vRight.x;
	vLeft.y += vRight.y;
	vLeft.z += vRight.z;
}

void operator-=(Vector3 & vLeft, Vector3 vRight)
{
	vLeft.x -= vRight.x;
	vLeft.y -= vRight.y;
	vLeft.z -= vRight.z;
}

void operator*=(Vector3 & v, float scale)
{
	v.x *= scale;
	v.y *= scale;
	v.z *= scale;
}

void operator/=(Vector3 & v, float scale)
{
	v.x /= scale;
	v.y /= scale;
	v.z /= scale;
}

Vector3 operator+ (Vector3 vLeft, Vector3 vRight)
{
	return Vector3(
		vLeft.x + vRight.x,
		vLeft.y + vRight.y,
		vLeft.z + vRight.z
	);
}

Vector3 operator- (Vector3 vLeft, Vector3 vRight)
{
	return Vector3(
		vLeft.x - vRight.x,
		vLeft.y - vRight.y,
		vLeft.z - vRight.z
	);
}

Vector3 operator* (Vector3 v, float f)
{
	return Vector3(
		v.x * f,
		v.y * f,
		v.z * f
	);
}

Vector3 operator* (float f, Vector3 v)
{
	return v * f;
}

ostream & operator<<(ostream & outputStream, Vector3 vector)
{
	// Bit of a hack for outputting as color from 0-255

	outputStream << int(vector.x * 255.99) << " " << int(vector.y * 255.99) << " " << int(vector.z * 255.99);
	return outputStream;
}

struct Sphere
{
	Sphere() = default;
	Sphere(Vector3 center, float radius)
		: center(center)
		, radius(radius)
	{ }

	Vector3 center;
	float radius;
};
struct Ray
{
	Ray() = default;

	Ray(Vector3 p0, Vector3 dir)
		: p0(p0)
		, dir(dir)
	{
		this->dir.normalize();
	}

	Vector3 pointAtT(float t)
	{
		return p0 + dir * t;
	}

	bool testSphere(const Sphere & sphere)
	{
		float a = dot(this->dir, this->dir);
		float b = 2 * (dot(this->dir, this->p0) - dot(this->dir, sphere.center));
		float c = dot(this->p0, this->p0) + dot(sphere.center, sphere.center) - 2 * dot(this->p0, sphere.center) - sphere.radius * sphere.radius;
		float discriminant = b * b - 4 * a * c;
		return discriminant > 0;
	}

	Vector3 color()
	{
		Sphere sphere(
			Vector3(0, 0, -1),
			0.5f);

		if (dir.z < -0.4)
		{
			bool brk = true;
		}

		if (dir.x < 0.1 && dir.x > -0.1 && dir.y < 0.1 && dir.y > -0.1)
		{
			bool brk = true;
		}

		if (testSphere(sphere))
			return Vector3(1, 0, 0);

		float t = 0.5f * (dir.y + 1);
		return (1 - t) * Vector3(1, 1, 1) + t * Vector3(0.5f, 0.7f, 1.0f);
	}

	Vector3 p0;
	Vector3 dir;
};

int main()
{
	constexpr int widthPixels = 200;
	constexpr int heightPixels = 100;

	ofstream file;
	file.open("output.ppm");
	file << "P3" << endl;
	file << widthPixels << " " << heightPixels << endl;
	file << 255 << endl;

	// Camera is at origin
	// Viewing plane is right-handed: +X right, +Y up, +Z out
	// Viewing plane is at Z = -1
	// Viewing plane spans X from -2 to 2
	// Viewing plane spans Y from -1 to 1

	Vector3 origin(0, 0, 0);
	Vector3 viewLowerLeftCorner(-2, -1, -1);
	Vector3 viewX(1, 0, 0);
	Vector3 viewY(0, 1, 0);
	float viewWidth = 4;
	float viewHeight = 2;

	for (int yPixel = 0; yPixel < heightPixels; yPixel++)
	{
		for (int xPixel = 0; xPixel < widthPixels; xPixel++)
		{
			if (yPixel == 50 && xPixel == 100)
			{
				bool brk = true;
			}

			float xPixelNorm = float(xPixel) / widthPixels;
			float yPixelNorm = 1 - (float(yPixel) / heightPixels);	// Subtract from 1 to get the value in view-space

			Ray r(
				origin,
				viewLowerLeftCorner + xPixelNorm * viewX * viewWidth + yPixelNorm * viewY * viewHeight);

			Vector3 colorOut = r.color();
			file << colorOut << endl;
		}
	}

	file.close();
	return 0;
}