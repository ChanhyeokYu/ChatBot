#include "MLP.h"

MLP::MLP(int input_size, int hidden_size, int output_size)
{
    W1.resize(hidden_size, vector<float>(input_size);
    b1.resize(hidden_size, 0.0f);
    z1.resize(hidden_size);
    a1.resize(hidden_size);

    W2.resize(output_size, vector<float>(hidden_size));
    b2.resize(output_size, 0.0f);
    z2.resize(output_size);
    output.resize(output_size);

    for (auto& row : W1)
    {
        for (auto& w : row)
        {
            w = randWeigth();
        }
    }

    for (auto& row : W2)
    {
        for (auto& w : row)
        {
            w = randWeigth();
        }
    }

}

std::vector<float> MLP::forward(const std::vector<float>& input)
{
    //Àº´ÐÃþ
    z1 = ::vector<float>(b1.size(), 0.0f);
    for (size_t i = 0; i < W1.size(); ++i)
    {
        for (size_t j = 0; j < input.size(); ++j)
        {
            z1[i] += W1[i][j] * input[j];
        }
    }

    for (size_t i = 0; i < z1.size(); ++i)
    {
        a1[i] = MachineMath::GetInstance().relu(z1[i] + b1[i]);
    }

    // Ãâ·ÂÃþ
    z2 = ::vector<float>(b2.size(), 0.0f);
    for (size_t i = 0; i < W2.size(); ++i)
    {
        for (size_t j = 0; j < a1.size(); ++j)
        {
            z2[i] += W2[i][j] * a1[j];
        }
    }
    
    for (size_t i = 0; i < z2.size(); ++i)
    {
        z1[i] += b2[i];
    }

    //softmax
    float max_z = *max_element(z2.begin(), z2.end());
    float sum = 0.0f;
    output = vector<float>(z2.size());
    for (size_t i = 0; i < z2.size(); ++i)
    {
        output[i] = exp(z2[i] - max_z);
        sum += output[i];
    }

    for (size_t i = 0; i < output.size(); ++i)
    {
        output[i] /= sum;
    }

    return output;
}

void MLP::backward(const std::vector<float>& input, const std::vector<float>& target, float learning_rate)
{
}

float MLP::randWeigth()
{
    return 0.0f;
}
