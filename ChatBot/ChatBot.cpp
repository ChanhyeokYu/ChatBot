// ChatBot.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <numeric>

using namespace std;

map<string, int> buildVocabulary(const vector<pair<string, string>> data);
vector<int> vectorize(const string& text, const map<string, int>& vocab);
string getResponse(const string& intent);

struct NaiveBayes
{
	map<string, int> labelCount;
	map<string, vector<int>> wordCount;
	map<string, double> prior;

	void train(const vector<pair<string, string>>& data, const map<string, int>& vocab)
	{
		for (const auto& [text, label] : data)
		{
			labelCount[label]++;
			if (wordCount.find(label) == wordCount.end())
			{
				wordCount[label] = vector<int>(vocab.size(), 0);
			}

			auto vec = vectorize(text, vocab);
			for (size_t i = 0; i < vec.size(); i++)
			{
				wordCount[label][i] += vec[i];
			}
		}

		int total = data.size();
		for (const auto& [label, count] : labelCount)
		{
			prior[label] = static_cast<double>(count) / total;
		}
	}

	string predict(const string& text, const map<string, int>& vocab)
	{
		auto vec = vectorize(text, vocab);
		double maxProb = -1e9;
		string bestLabel;

		for (const auto& [label, countVec] : wordCount)
		{
			double logProb = log(prior[label] + 1e-10);

			for (size_t i = 0; i < vec.size(); i++)
			{
				int wordFreq = countVec[i] + 1;
				int totalWords = accumulate(countVec.begin(), countVec.end(), 0) + vocab.size();
				logProb += vec[i] * log(static_cast<double>(wordFreq) / totalWords);
			}

			if (logProb > maxProb)
			{
				maxProb = logProb;
				bestLabel = label;
			}
		}
		return bestLabel;
	}

};

vector<pair<string, string>> dataset = {
	{"안녕", "greeting"},
	{"안녕하세요", "greeting"},
	{"비 와?", "weather"},
	{"날씨 어떄?", "weather"},
	{"이름이 뭐야?", "name"},
	{"너 이름은?", "name"},
};

map<string, int> buildVocabulary(const vector<pair<string, string>> data)
{
	map<string, int> vocab;
	int index = 0;
	for (const auto& [text, intent] : data)
	{
		istringstream iss(text);
		string word;
			while (iss >> word)
			{
				if (vocab.find(word) == vocab.end())
				{
					vocab[word] = index++;
				}
			}
	}
	return vocab;

}

vector<int> vectorize(const string& text, const map<string, int>& vocab)
{
	vector<int> vec(vocab.size(), 0);
	istringstream iss(text);
	string word;
	while (iss >> word)
	{
		if (vocab.find(word) != vocab.end())
		{
			vec[vocab.at(word)] += 1;
		}
	}
	return vec;
}

string getResponse(const string& intent)
{
	
	if (intent == "greeting") return "안녕하세요";
	if (intent == "weather") return "날씨는 제가 잘 몰라요 ㅠㅠ";
	if (intent == "name") return "저는 챗봇이에요!";
	return "무슨 말인지 잘 모르겠어요.";

}

int main()
{
	vector<pair<string, string>> data = {
		{"안녕", "greeting"}, {"안녕하세요", "greeting"},
		{"비 와?", "weather"}, {"날씨 어때?", "weather"},
		{"이름이 뭐야?", "name"}, {"너 이름은?", "name"}
	};

	auto vocab = buildVocabulary(data);

	NaiveBayes nb;
	nb.train(data, vocab);

	while (true)
	{
		string input;
		cout << "당신: ";
		getline(cin, input);
		if (input == "종료")
		{
			break;
		}

		string intent = nb.predict(input, vocab);
		cout << "챗봇: " << getResponse(intent) << endl;
	}

	return 0;

}

