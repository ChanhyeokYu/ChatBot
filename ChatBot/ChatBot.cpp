// ChatBot.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>

#include "MLP.h"

using namespace std;

int main()
{
	MLP* mlp = new MLP(5,5,5);
	
	vector<float> v1 = { 1.0f };
	mlp->forward(v1);
	return 0;

}

