#include <fstream>
#include <math.h>

#include "main.h"

using namespace std;

float Vector3::length() const
{
	return sqrt(lengthSquared());
}

float Vector3::lengthSquared() const
{
	return x * x + y * y + z * z;
}

void Vector3::normalizeInPlace()
{
	float len = length();
	x /= len;
	y /= len;
	z /= len;
}

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

Vector3 normalize(const Vector3 & v)
{
	Vector3 result = v;
	float len = v.length();
	result.x /= len;
	result.y /= len;
	result.z /= len;

	return result;
}

Vector3 operator+(const Vector3 & v)
{
	return Vector3(v.x, v.y, v.z);
}

Vector3 operator-(const Vector3 & v)
{
	return Vector3(-v.x, -v.y, -v.z);
}

void operator+=(Vector3 & vLeft, const Vector3 & vRight)
{
	vLeft.x += vRight.x;
	vLeft.y += vRight.y;
	vLeft.z += vRight.z;
}

void operator-=(Vector3 & vLeft, const Vector3 & vRight)
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

Vector3 operator+ (const Vector3 & vLeft, const Vector3 & vRight)
{
	return Vector3(
		vLeft.x + vRight.x,
		vLeft.y + vRight.y,
		vLeft.z + vRight.z
	);
}

Vector3 operator- (const Vector3 & vLeft, const Vector3 & vRight)
{
	return Vector3(
		vLeft.x - vRight.x,
		vLeft.y - vRight.y,
		vLeft.z - vRight.z
	);
}

Vector3 hadamard(const Vector3 & vLeft, const Vector3 & vRight)
{
	return Vector3(
		vLeft.x * vRight.x,
		vLeft.y * vRight.y,
		vLeft.z * vRight.z
	);
}

Vector3 operator* (const Vector3 & v, float f)
{
	return Vector3(
		v.x * f,
		v.y * f,
		v.z * f
	);
}

Vector3 operator* (float f, const Vector3 & v)
{
	return v * f;
}

Vector3 operator/(const Vector3 & v, float f)
{
	return Vector3(
		v.x / f,
		v.y / f,
		v.z / f
	);
}

ostream & operator<<(ostream & outputStream, const Vector3 & vector)
{
	// Bit of a hack for outputting as color from 0-255

	outputStream << int(vector.x * 255.99) << " " << int(vector.y * 255.99) << " " << int(vector.z * 255.99);
	return outputStream;
}

Vector3 randomVectorInsideUnitSphere()
{
    // @Slow
    // Better implementations exist, but this one is the easiest!

    Vector3 result;
    while (true)
    {
        // Random vector where x, y, z are [0-1)
        
        result = Vector3(rand0Incl1Excl(), rand0Incl1Excl(), rand0Incl1Excl());

        // Transform x, y, z to be [-1, 1)
        
        result *= 2;
        result -= Vector3(1, 1, 1);

        // Only accept if inside unit sphere

        if (result.lengthSquared() <= 1.0f)
            break;
    }

    return result;
}

Vector3 randomVectorInsideUnitDisk()
{
    // @Slow
    // Better implementations exist, but this one is the easiest!

    Vector3 result;
    while (true)
    {
        
        result = Vector3(rand0Incl1Excl() * 2 - 1, rand0Incl1Excl() * 2 - 1, 0);

        // Only accept if inside unit disk

        if (result.lengthSquared() <= 1.0f)
            break;
    }

    return result;
}

Vector3 reflect(const Vector3 & v, const Vector3 & n)
{
    Vector3 result = v - 2 * dot(v, n) * n;
    return result;
}

bool refractWithSchlickProbability(const Vector3 & v, const Vector3 & n, float refractionIndexFrom, float refractionIndexTo, Vector3 * vecOut)
{
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel


    Vector3 nAdjusted = n;
    if (dot(v, n) > 0)
    {
        nAdjusted = -n;
    }
    
    float cosThetaIncident = dot(-v, nAdjusted);

	// Reject due to Schlick probability

	{
		float reflectProb = (1 - refractionIndexTo) / (1 + refractionIndexTo);
		reflectProb = reflectProb * reflectProb;
		reflectProb = reflectProb + (1 - reflectProb) * pow(1 - cosThetaIncident, 5);

		if (rand0Incl1Excl() < reflectProb)
			return false;
	}

    float refractionRatio = refractionIndexFrom / refractionIndexTo;
    float k = 1 - refractionRatio * refractionRatio * (1 - cosThetaIncident * cosThetaIncident);

	// Reject due to total internal reflection

	if (k < 0)
		return false;
    
    *vecOut = refractionRatio * v + nAdjusted * (refractionRatio * cosThetaIncident - sqrtf(k));
	return true;
}

float rand0Incl1Excl()
{
	return rand() / (RAND_MAX + 1.0f);
}

bool Sphere::testHit(const Ray & ray, float tMin, float tMax, HitRecord * hitOut) const
{
	Vector3 center = posAtTime(ray.time);

	float a = dot(ray.dir, ray.dir);
	float b = 2 * (dot(ray.dir, ray.p0) - dot(ray.dir, center));
	float c = dot(ray.p0, ray.p0) + dot(center, center) - 2 * dot(ray.p0, center) - this->radius * this->radius;

	float discriminant = b * b - 4 * a * c;
	if (discriminant < 0)
		return false;

	float t0 = (-b - sqrt(discriminant)) / 2 * a;
	float t1 = (-b + sqrt(discriminant)) / 2 * a;

	if (t0 > tMin && t0 < tMax)
	{
		hitOut->t = t0;
		hitOut->normal = normalize(ray.pointAtT(t0) - center);
        hitOut->hittable = this;
		return true;
	}
	else if (t1 > tMin && t1 < tMax)
	{
		hitOut->t = t1;
		hitOut->normal = normalize(ray.pointAtT(t1) - center);
        hitOut->hittable = this;
		return true;
	}
	else
	{
		return false;
	}
}

bool Sphere::tryComputeBoundingBox(float t0, float t1, Aabb * aabbOut) const
{
	float dT = t1 - t0;
	Vector3 p1Center = p0Center + dT * velocity;

	Vector3 vecRadius = Vector3(radius, radius, radius);

	Aabb aabb0(p0Center + vecRadius, p0Center - vecRadius);
	Aabb aabb1(p1Center + vecRadius, p0Center - vecRadius);

	*aabbOut = Aabb(aabb0, aabb1);
    return true;
}

Vector3 Sphere::posAtTime(float time) const
{
	return p0Center + velocity * time;
}

Vector3 Ray::pointAtT(float t) const
{
	return p0 + dir * t;
}

Vector3 Ray::color(BvhNode * bvhNode, int rayDepth) const
{
    constexpr int MAX_RAY_DEPTH = 50;
    if (rayDepth > MAX_RAY_DEPTH)
    {
        return Vector3(0, 0, 0);
    }
    
	float tClosest = FLT_MAX;
	HitRecord hit;
	bool hitSuccess = bvhNode->testHit(*this, 0.0001f, FLT_MAX, &hit);

	if (hitSuccess)
	{
        Vector3 attenuation;
        Ray rayScattered;
        
        if (hit.hittable->material->scatter(*this, hit, &attenuation, &rayScattered))
        {
            return hadamard(attenuation, rayScattered.color(bvhNode, rayDepth + 1));
        }
        else
        {
            return Vector3(0, 0, 0);
        }
	}
	else
	{
		// blue-white "sky" lerp

		float t = 0.5f * (dir.y + 1);
		return (1 - t) * Vector3(1, 1, 1) + t * Vector3(0.5f, 0.7f, 1.0f);
	}
}

bool LambertianMaterial::scatter(const Ray & ray, const HitRecord & hitRecord, Vector3 * attenuationOut, Ray * rayScatteredOut) const
{
    if (dot(ray.dir, hitRecord.normal) > 0)
        return false;
    
    Vector3 hitPos = ray.pointAtT(hitRecord.t);
    Vector3 probeUnitSphereCenter = hitPos + hitRecord.normal;
    *rayScatteredOut = Ray(hitPos, probeUnitSphereCenter + randomVectorInsideUnitSphere() - hitPos, ray.time);
    *attenuationOut = albedo;
    return true;
}

bool MetalMaterial::scatter(const Ray & ray, const HitRecord & hitRecord, Vector3 * attenuationOut, Ray * rayScatteredOut) const
{
    if (dot(ray.dir, hitRecord.normal) > 0)
        return false;
    
    Vector3 hitPos = ray.pointAtT(hitRecord.t);
    *rayScatteredOut = Ray(hitPos, reflect(ray.dir, hitRecord.normal) + fuzziness * randomVectorInsideUnitSphere(), ray.time);
    *attenuationOut = albedo;
    return true;
}

bool DielectricMaterial::scatter(const Ray & ray, const HitRecord & hitRecord, Vector3 * attenuationOut, Ray * rayScatteredOut) const
{
	*attenuationOut = Vector3(1.0, 1.0, 1.0);

	float isInside = dot(ray.dir, hitRecord.normal) > 0;

	float refractiveIndexFrom	=	isInside	?	this->refractiveIndex	:	1.0f;
	float refractiveIndexTo		=	isInside	?	1.0f					:	this->refractiveIndex;

	Vector3 refracted;
	if (refractWithSchlickProbability(ray.dir, isInside ? -hitRecord.normal : hitRecord.normal, refractiveIndexFrom, refractiveIndexTo, &refracted))
	{
		rayScatteredOut->p0 = ray.pointAtT(hitRecord.t);
		rayScatteredOut->dir = refracted;
	}
	else
	{
		rayScatteredOut->p0 = ray.pointAtT(hitRecord.t);
		rayScatteredOut->dir = reflect(ray.dir, hitRecord.normal);
	}

	return true;
}

Camera::Camera(Vector3 pos, Vector3 lookAt, float fovDeg, float aspectRatio, float lensRadius, float time0, float time1)
    : pos(pos)
	, fovDeg(fovDeg)
	, aspectRatio(aspectRatio)
	, lensRadius(lensRadius)
	, time0(time0)
	, time1(time1)
{
    this->forward = normalize(lookAt - pos);
    this->right = normalize(cross(this->forward, Vector3(0, 1, 0)));	// NOTE: Assuming <0, 1, 0> is the "up vector" that most camera API's ask for.
    this->up = normalize(cross(this->right, this->forward));

    float fovRad = fovDeg * 3.14159f / 180;
    
	// halfUnitH and halfUnitW are in a space where the plane is 1 unit in front of the camera (before it gets scaled up by focusDistance)

    float halfUnitH = tan(fovRad / 2);
    float halfUnitW = halfUnitH * aspectRatio;

	float focusDistance = (lookAt - pos).length();
    this->botLeftViewPosCached = this->pos + (this->forward - this->right * halfUnitW - this->up * halfUnitH) * focusDistance;
    this->topRightViewPosCached = this->pos + (this->forward + this->right * halfUnitW + this->up * halfUnitH) * focusDistance;
    this->wCached = halfUnitW * 2 * focusDistance;
    this->hCached = halfUnitH * 2 * focusDistance;
}

Ray Camera::rayAt(float s, float t)
{
	Vector3 p0 = this->pos + this->lensRadius * randomVectorInsideUnitDisk();
    Ray result(
        p0,
        (this->botLeftViewPosCached + s * this->wCached * this->right + t * this->hCached * this->up) - p0,
		lerp(time0, time1, rand0Incl1Excl()));

    return result;
}

int main()
{
	constexpr int widthPixels = 400;
	constexpr int heightPixels = 200;
	constexpr float fov = 20.0f;
    constexpr float aspectRatio = widthPixels / float(heightPixels);
	constexpr float lensRadius = 0.05f;
	constexpr int COUNT_SAMPLES_PER_PIXEL = 100;
	constexpr float time0 = 0;
	constexpr float time1 = 1;

	ofstream file;
	file.open("output.ppm");
	file << "P3" << endl;
	file << widthPixels << " " << heightPixels << endl;
	file << 255 << endl;

	constexpr int MAX_COUNT_HITTABLES = 512;
	IHittable * hittables[MAX_COUNT_HITTABLES];

	int countHittables = 0;

	// The code to generate this scene is almost entirely ripped from the text. I'm not super interested in how the scene is generated... just in the rendering of it!

	{
		// Ground sphere

		hittables[countHittables++] =
			new Sphere(
				Vector3(0, -1000, 0),
				1000,
				new LambertianMaterial(
					Vector3(0.5f, 0.5f, 0.5f)),
				kZero);

		// Little spheres

		int i = 1;
		for (int a = -11; a < 11; a++)
		{
			for (int b = -11; b < 11; b++)
			{
				auto choose_mat = rand0Incl1Excl();
				float radius = 0.2f;
				Vector3 center(a + 0.9f * rand0Incl1Excl(), 0.2f, b + 0.9f * rand0Incl1Excl());

				if ((center - Vector3(4, 0.2f, 0)).length() > 1.8f)
				{
					if (choose_mat < 0.8f)
					{
						// diffuse
						Vector3 vel(0, rand0Incl1Excl() * 0.5f, 0);
						auto albedo = Vector3(rand0Incl1Excl(), rand0Incl1Excl(), rand0Incl1Excl());
						hittables[countHittables++] = new Sphere(center, radius, new LambertianMaterial(albedo), vel);
					}
					else if (choose_mat < 0.95f)
					{
						// metal
						auto albedo = Vector3(rand0Incl1Excl(), rand0Incl1Excl(), rand0Incl1Excl());
						albedo = albedo / 2.0f + Vector3(0.5f, 0.5f, 0.5f);
						auto fuzz = rand0Incl1Excl() / 2.0f;
						hittables[countHittables++] = new Sphere(center, radius, new MetalMaterial(albedo, fuzz), kZero);
					}
					else
					{
						// glass
						hittables[countHittables++] = new Sphere(center, radius, new DielectricMaterial(1.5f), kZero);
					}
				}
			}
		}

		// Big spheres

		hittables[countHittables++] = new Sphere(Vector3(0, 1, 0), 1.0f, new DielectricMaterial(1.5f), kZero);
		hittables[countHittables++] = new Sphere(Vector3(-4, 1, 0), 1.0f, new LambertianMaterial(Vector3(0.4f, 0.2f, 0.1f)), kZero);
		hittables[countHittables++] = new Sphere(Vector3(4, 1, 0), 1.0f, new MetalMaterial(Vector3(0.7f, 0.6f, 0.5f), 0.0f), kZero);
	}

    Camera cam(
        Vector3(13, 2, 3),
        Vector3(0, 0, 0),
        fov,
		aspectRatio,
		lensRadius,
		time0,
		time1);

	BvhNode * bvh = buildBvh(hittables, countHittables, time0, time1);

	for (int yPixel = 0; yPixel < heightPixels; yPixel++)
	{
		for (int xPixel = 0; xPixel < widthPixels; xPixel++)
		{
			Vector3 colorCumulative;

			for (int iSample = 0; iSample < COUNT_SAMPLES_PER_PIXEL; iSample++)
			{
				float perturbX = rand0Incl1Excl();
				float perturbY = rand0Incl1Excl();

				float xViewNormalized = (xPixel + perturbX) / widthPixels;
				float yViewNormalized = 1 - ((yPixel + perturbY) / heightPixels);	// Subtract from 1 to get the value in view-space

				Ray ray = cam.rayAt(xViewNormalized, yViewNormalized);
				colorCumulative += ray.color(bvh, 0);
			}

			Vector3 colorOut = colorCumulative / COUNT_SAMPLES_PER_PIXEL;

            // Gamma correct

            colorOut.x = sqrt(colorOut.x);
            colorOut.y = sqrt(colorOut.y);
            colorOut.z = sqrt(colorOut.z);
            
			file << colorOut << endl;
		}
	}

	file.close();
	return 0;
}

bool BvhNode::testHit(const Ray & ray, float tMin, float tMax, HitRecord * hitOut) const
{
    if (!aabb.testHit(ray, tMin, tMax))
        return false;

    // This relies on testHit not modifying the contents of hitOut if it returns false
    
    bool hitLeft = left->testHit(ray, tMin, tMax, hitOut);
    bool hitRight = right->testHit(ray, tMin, hitLeft ? hitOut->t : tMax, hitOut);

    return hitLeft || hitRight;
}

bool BvhNode::tryComputeBoundingBox(float t0, float t1, Aabb * aabbOut) const
{
    *aabbOut = aabb;
    return true;
}

BvhNode::BvhNode(IHittable * left, IHittable * right, float time0, float time1)
    : IHittable(nullptr)
	, left(left)
    , right(right)
{
    Aabb aabbLeft;
    Aabb aabbRight;
    
    Verify(left->tryComputeBoundingBox(time0, time1, &aabbLeft));
    Verify(right->tryComputeBoundingBox(time0, time1, &aabbRight));

    aabb = Aabb(aabbLeft, aabbRight);
}

static int compareByAxis(IHittable * const& h0, IHittable * const& h1, int axis)
{
    Aabb aabb0;
    Aabb aabb1;

    // No need to use real times here, as this function is only used to
    // create the BVH heirarchy, and choosing left vs right for a given
    // hittable doesn't affect correctness.
    
    Verify(h0->tryComputeBoundingBox(0, 0, &aabb0));
    Verify(h1->tryComputeBoundingBox(0, 0, &aabb1));

    return aabb0.min.elements[axis] < aabb1.min.elements[axis] ? -1 : aabb0.min.elements[axis] == aabb1.min.elements[axis] ? 0 : 1;
}

static int compareByX(IHittable * const& h0, IHittable * const& h1) { return compareByAxis(h0, h1, 0); }
static int compareByY(IHittable * const& h0, IHittable * const& h1) { return compareByAxis(h0, h1, 1); }
static int compareByZ(IHittable * const& h0, IHittable * const& h1) { return compareByAxis(h0, h1, 2); }

BvhNode * buildBvh(IHittable ** aHittable, int cHittable, float time0, float time1)
{
    Assert(cHittable > 2);

    if (cHittable == 2)
    {
        return new BvhNode(*aHittable, *(aHittable + 1), time0, time1);
    }
    else
    {
        int axis = rand() % 3;
        bubbleSort(
            aHittable,
            cHittable,
            axis == 0 ? compareByX : axis == 1 ? compareByY : compareByZ);

        int cLeft = cHittable / 2;
        int cRight = cHittable - cLeft;

        IHittable * left = nullptr;
        if (cLeft == 1)
        {
            left = *aHittable;
        }
        else
        {
            Assert(cLeft > 1);
            left = buildBvh(aHittable, cLeft, time0, time1);
        }

        IHittable * right = nullptr;
        if (cRight == 1)
        {
            right = *(aHittable + cLeft);
        }
        else
        {
            Assert(cRight > 1);
            right = buildBvh(aHittable + cLeft, cRight, time0, time1);
        }
        
        return new BvhNode(left, right, time0, time1); 
    }
}

Aabb::Aabb(Vector3 p0, Vector3 p1)
{
	min.x = fmin(p0.x, p1.x);
	min.y = fmin(p0.y, p1.y);
	min.z = fmin(p0.z, p1.z);

	max.x = fmax(p0.x, p1.x);
	max.y = fmax(p0.y, p1.y);
	max.z = fmax(p0.z, p1.z);
}

Aabb::Aabb(Aabb aabb0, Aabb aabb1)
{
	min.x = fmin(aabb0.min.x, aabb1.min.x);
    min.y = fmin(aabb0.min.y, aabb1.min.y);
    min.z = fmin(aabb0.min.z, aabb1.min.z);

	max.x = fmax(aabb0.max.x, aabb1.max.x);
    max.y = fmax(aabb0.max.y, aabb1.max.y);
    max.z = fmax(aabb0.max.z, aabb1.max.z);
}

bool Aabb::testHit(const Ray & ray, float tMin, float tMax) const
{
	for (int axis = 0; axis < 3; axis++)
	{
		float invDirection = 1.0f / ray.dir.elements[axis];
		float deltaDir0 = min.elements[axis] - ray.p0.elements[axis];
		float deltaDir1 = max.elements[axis] - ray.p0.elements[axis];

		float t0 = deltaDir0 * invDirection;
		float t1 = deltaDir1 * invDirection;

		// If the ray's direction in this axis is 0, t0 and t1 will both be + or - infinity if the ray origin
		//	is outside the AABB and will be opposite signed infinities if inside.

		// If the ray origin lies on this axis of the AABB and it's direction in this axis is 0, we will get
		//	NaN for either t0 or t1, and either + or - infinity for the other. In this case, we may return
		//	either true or false. We could do extra work to detect and make this scenario consistent (see
		//	link), but in practice it should not matter:
		//	https://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/

		// Update window that next axis is allowed to hit in

		tMin = fmax(tMin, fmin(t0, t1));
		tMax = fmin(tMax, fmax(t0, t1));
	}

	return tMax > fmax(tMin, 0);
}
