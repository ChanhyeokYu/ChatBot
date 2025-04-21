#pragma once
#include "MachineMath.h"

class MLP
{
public:
	MLP(int input_size, int hidden_size, int output_size);

	std::vector<float> forward(const std::vector<float>& input);
	void backward(const std::vector<float>& input, const std::vector<float>& target, float learning_rate);

private:
	std::vector<std::vector<float>> W1, W2; // 입력층 -> 은닉층
	std::vector<float> b1, b2; // 은닉층 -> 출력층
	std::vector<float> z1, a1, z2, output; // 은닉층 결과(z는 선형, a는 활성화, z2,output는 출력층 결과)

	float randWeigth();

};

