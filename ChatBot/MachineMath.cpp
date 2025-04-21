#include "MachineMath.h"

float MachineMath::sigmoid(float x)
{
	return 1.0f / (1.0f + ::exp(-x));
}

float MachineMath::relu(float x)
{
	return ::max(0.0f, x);
}

float MachineMath::relu_derivative(float x)
{
	return x > 0 ? 1.0f : 0.0f;
}

float MachineMath::cross_entropy_loss(const::vector<float>& y_hat, const::vector<float>& y)
{
	float loss = 0.0f;
	for (size_t i = 0; i < y.size(); i++)
	{
		loss -= y[i] * ::log(y_hat[i] + 1e-9);
	}

	return loss;
}
