#pragma once
#include "MachineMath.h"

class MLP
{
public:
	MLP(int input_size, int hidden_size, int output_size);

	std::vector<float> forward(const std::vector<float>& input);
	void backward(const std::vector<float>& input, const std::vector<float>& target, float learning_rate);

private:
	std::vector<std::vector<float>> W1, W2; // �Է��� -> ������
	std::vector<float> b1, b2; // ������ -> �����
	std::vector<float> z1, a1, z2, output; // ������ ���(z�� ����, a�� Ȱ��ȭ, z2,output�� ����� ���)

	float randWeigth();

};

