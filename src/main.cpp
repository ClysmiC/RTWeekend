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
	float a = dot(ray.dir, ray.dir);
	float b = 2 * (dot(ray.dir, ray.p0) - dot(ray.dir, this->center));
	float c = dot(ray.p0, ray.p0) + dot(this->center, this->center) - 2 * dot(ray.p0, this->center) - this->radius * this->radius;

	float discriminant = b * b - 4 * a * c;
	if (discriminant < 0)
		return false;

	float t0 = (-b - sqrt(discriminant)) / 2 * a;
	float t1 = (-b + sqrt(discriminant)) / 2 * a;

	if (t0 > tMin && t0 < tMax)
	{
		hitOut->t = t0;
		hitOut->normal = normalize(ray.pointAtT(t0) - this->center);
        hitOut->hitable = this;
		return true;
	}
	else if (t1 > tMin && t1 < tMax)
	{
		hitOut->t = t1;
		hitOut->normal = normalize(ray.pointAtT(t1) - this->center);
        hitOut->hitable = this;
		return true;
	}
	else
	{
		return false;
	}
}

Vector3 Ray::pointAtT(float t) const
{
	return p0 + dir * t;
}

Vector3 Ray::color(IHitable ** aHitable, int cHitable, int rayDepth) const
{
    constexpr int MAX_RAY_DEPTH = 50;
    if (rayDepth > MAX_RAY_DEPTH)
    {
        return Vector3(0, 0, 0);
    }
    
	float tClosest = FLT_MAX;
	HitRecord hitClosest;

	for (int iHitable = 0; iHitable < cHitable; iHitable++)
	{
		IHitable * pHitable = *(aHitable + iHitable);

		HitRecord hit;
		if (pHitable->testHit(*this, 0.0001f, tClosest, &hit))
		{
			hitClosest = hit;
			tClosest = hit.t;
		}
	}

	bool hit = tClosest < FLT_MAX;
	if (hit)
	{
        Vector3 attenuation;
        Ray rayScattered;
        
        if (hitClosest.hitable->material->scatter(*this, hitClosest, &attenuation, &rayScattered))
        {
            return hadamard(attenuation, rayScattered.color(aHitable, cHitable, rayDepth + 1));
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
    *rayScatteredOut = Ray(hitPos, probeUnitSphereCenter + randomVectorInsideUnitSphere() - hitPos);
    *attenuationOut = albedo;
    return true;
}

bool MetalMaterial::scatter(const Ray & ray, const HitRecord & hitRecord, Vector3 * attenuationOut, Ray * rayScatteredOut) const
{
    if (dot(ray.dir, hitRecord.normal) > 0)
        return false;
    
    Vector3 hitPos = ray.pointAtT(hitRecord.t);
    *rayScatteredOut = Ray(hitPos, reflect(ray.dir, hitRecord.normal) + fuzziness * randomVectorInsideUnitSphere());
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

Camera::Camera(Vector3 pos, Vector3 lookAt, float fovDeg, float aspectRatio, float lensRadius)
    : pos(pos), fovDeg(fovDeg), aspectRatio(aspectRatio), lensRadius(lensRadius)
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
        (this->botLeftViewPosCached + s * this->wCached * this->right + t * this->hCached * this->up) - p0);

    return result;
}

int main()
{
	constexpr int widthPixels = 800;
	constexpr int heightPixels = 400;
	constexpr float fov = 20;
    constexpr float aspectRatio = widthPixels / float(heightPixels);
	constexpr float lensRadius = 0.05;	// Is this way too big?
	constexpr int COUNT_SAMPLES_PER_PIXEL = 200;

	ofstream file;
	file.open("output.ppm");
	file << "P3" << endl;
	file << widthPixels << " " << heightPixels << endl;
	file << 255 << endl;

	constexpr int MAX_COUNT_HITABLES = 512;
	IHitable * hitables[MAX_COUNT_HITABLES];

	int countHitables = 0;

	// The code to generate this scene is almost entirely ripped from the text. I'm not super interested in how the scene is generated... just in the rendering of it!

	{
		hitables[countHitables++] = new Sphere(Vector3(0, -1000, 0), 1000, new LambertianMaterial(Vector3(0.5f, 0.5f, 0.5f)));

		int i = 1;
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				auto choose_mat = rand0Incl1Excl();
				Vector3 center(a + 0.9 * rand0Incl1Excl(), 0.2, b + 0.9 * rand0Incl1Excl());
				if ((center - Vector3(4, 0.2, 0)).length() > 0.9) {
					if (choose_mat < 0.8) {
						// diffuse
						auto albedo = Vector3(rand0Incl1Excl(), rand0Incl1Excl(), rand0Incl1Excl());
						hitables[countHitables++] = new Sphere(center, 0.2, new LambertianMaterial(albedo));
					}
					else if (choose_mat < 0.95) {
						// metal
						auto albedo = Vector3(rand0Incl1Excl(), rand0Incl1Excl(), rand0Incl1Excl());
						albedo = albedo / 2.0f + Vector3(0.5, 0.5, 0.5);
						auto fuzz = rand0Incl1Excl() / 2.0f;
						hitables[countHitables++] = new Sphere(center, 0.2, new MetalMaterial(albedo, fuzz));
					}
					else {
						// glass
						hitables[countHitables++] = new Sphere(center, 0.2, new DielectricMaterial(1.5));
					}
				}
			}
		}

		hitables[countHitables++] = new Sphere(Vector3(0, 1, 0), 1.0, new DielectricMaterial(1.5));
		hitables[countHitables++] = new Sphere(Vector3(-4, 1, 0), 1.0, new LambertianMaterial(Vector3(0.4, 0.2, 0.1)));
		hitables[countHitables++] = new Sphere(Vector3(4, 1, 0), 1.0, new MetalMaterial(Vector3(0.7, 0.6, 0.5), 0.0));
	}

    Camera cam(
        Vector3(13, 2, 3),
        Vector3(0, 0, 0),
        fov,
		aspectRatio,
		lensRadius);

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
				colorCumulative += ray.color(hitables, countHitables, 0);
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
