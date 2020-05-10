#pragma once

#include <ostream>

struct IHitable;

struct Vector3
{
	float x;
	float y;
	float z;

	Vector3() : x(0), y(0), z(0) { }
	Vector3(float x, float y, float z) : x(x), y(y), z(z) { }

	float length() const;
	float lengthSquared() const;
	void normalizeInPlace();
};

float dot(const Vector3 & lhs, const Vector3 & rhs);
Vector3 cross(const Vector3 & lhs, const Vector3 & rhs);
Vector3 normalize(const Vector3 & v);
Vector3 operator+(Vector3 v);
Vector3 operator-(Vector3 v);
void operator+=(Vector3 & vLeft, Vector3 vRight);
void operator-=(Vector3 & vLeft, Vector3 vRight);
void operator*=(Vector3 & v, float scale);
void operator/=(Vector3 & v, float scale);
Vector3 operator+ (Vector3 vLeft, Vector3 vRight);
Vector3 operator- (Vector3 vLeft, Vector3 vRight);
Vector3 operator* (Vector3 v, float f);
Vector3 operator* (float f, Vector3 v);
Vector3 operator/ (Vector3 v, float f);
std::ostream & operator<<(std::ostream & outputStream, Vector3 vector);
Vector3 randomVectorInsideUnitSphere();

struct Ray
{
	Vector3 p0;
	Vector3 dir;

	Ray(Vector3 p0, Vector3 dir) : p0(p0), dir(normalize(dir)) { }

	Vector3 pointAtT(float t) const;
	Vector3 color(IHitable ** aHitable, int cHitable) const;
};

struct HitRecord
{
	float t;
	Vector3 normal;

	HitRecord() : t(0), normal() {}
};

struct IHitable
{
	virtual bool testHit(const Ray & ray, float tMin, float tMax, HitRecord * hitOut) const = 0;
};

struct Sphere : public IHitable
{
	Vector3 center;
	float radius;

	Sphere(Vector3 center, float radius) : center(center), radius(radius) { }

	bool testHit(const Ray & ray, float tMin, float tMax, HitRecord * hitOut) const override;
};

float rand0Incl1Excl();
