#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

int main(int argc, char** argv)
{
    // Read the data from csv file
    cv::Ptr<cv::ml::TrainData> cvml = cv::ml::TrainData::loadFromCSV(
        std::string("char_datasetNM2.csv"), 0, 0);

    // Select 80% for the training 
    cvml->setTrainTestSplitRatio(0.8, true);

    cv::Ptr<cv::ml::Boost> boost;
    std::ifstream ifile("./trained_classifierNM2.xml");
    if (ifile)
    {
        //The file exists, so we don't want to train 
        printf("Found trained_boost_char.xml file, remove it if you want to retrain with new data ... \n");
        boost = cv::ml::StatModel::load<cv::ml::Boost>("./trained_classifierNM2.xml");
    }
    else {
        //Train with 100 features
        printf("Training ... \n");
        boost = cv::ml::Boost::create();
        boost->setBoostType(cv::ml::Boost::REAL);
        boost->setWeakCount(100);
        boost->setWeightTrimRate(0.0);
        boost->setMaxDepth(1);
        boost->setUseSurrogates(false);

        bool ok = boost->train(cvml);
    }

    // Calculate the test and train errors
    cv::Mat train_responses, test_responses;
    float fl1 = boost->calcError(cvml, false, train_responses);
    float fl2 = boost->calcError(cvml, true, test_responses);
    printf("Error train %f \n", fl1);
    printf("Error test %f \n", fl2);

    // Try a char
    cv::Mat sample = (cv::Mat_<float>(1, 7) << 0.870690, 0.096485, 2.000000, 2.000000, 0.137080, 1.269940, 2.000000);
    float prediction = boost->predict(sample, cv::noArray(), 0);
    float votes = boost->predict(sample, cv::noArray(), 
        cv::ml::DTrees::PREDICT_SUM | cv::ml::StatModel::RAW_OUTPUT);

    printf("\n The char sample is predicted as: %f (with number of votes = %f)\n",
        prediction, votes);
    printf(" Class probability (using Logistic Correction) is P(r|character) = %f\n",
        1.0f - 1.0f / (1.0f + std::exp(-2.0f * votes)));

    // Try a NONchar
    // static const float arr2[] = {0,1.500000,0.072162,0.000000,8.000000,0.188095,1.578947,16.000000};
    cv::Mat sample2 = (cv::Mat_<float>(1, 7) << 0.565217, 0.103749, 1.000000, 2.000000, 0.032258, 1.525692, 10.000000);
    prediction = boost->predict(sample2, cv::noArray(), 0);
    votes = boost->predict(sample2, cv::noArray(), 
        cv::ml::DTrees::PREDICT_SUM | cv::ml::StatModel::RAW_OUTPUT);

    printf("\n The non_char sample is predicted as: %f (with number of votes = %f)\n",
        prediction, votes);
    printf(" Class probability (using Logistic Correction) is P(r|character) = %f\n\n",
        1.0f - 1.0f / (1.0f + std::exp(-2.0f * votes)));

    // Save the trained classifier
    boost->save(std::string("./trained_classifierNM2.xml"));

    return EXIT_SUCCESS;
}
