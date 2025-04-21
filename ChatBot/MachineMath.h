#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <string>
#include <numeric>

using namespace std;

class MachineMath
{
public:
	static MachineMath& GetInstance()
	{
		static MachineMath instance_;
		return instance_;
	}

public:
	float sigmoid(float x);
	float relu(float x);
	float relu_derivative(float x);
	float cross_entropy_loss(const ::vector<float>& y_hat, const ::vector<float>& y);

private:
	MachineMath();
	MachineMath(const MachineMath&&) = delete;	
	MachineMath(const MachineMath&) = delete;
	MachineMath& operator=(MachineMath&) = delete;
	~MachineMath() = default;

};

