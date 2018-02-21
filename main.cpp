#include <caffe/caffe.hpp>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <fstream>
#include "caffe/layers/memory_data_layer.hpp"

using namespace std;
using namespace cv;

template<class type>
void Lsb2Msb(type &a)
{
    char *x = (char *)&a;
    int nb = sizeof(type);
    for (int i=0;i<nb/2;i++)
        swap(x[i], x[nb-i-1]);
}

void MnistToMat(char *nameData, char *nameLabel, vector<Mat> &trainImages, std::vector<uint> &labels, bool display = false);
void VectorMat2VectorFloat(vector<Mat> img, vector<uint> lab, vector<float> &data, vector<float> &label);

int main()
{
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    vector<Mat> trainImages;
    vector<uint> trainLabels;
    vector<Mat> testImages;
    vector<uint> testLabels;
    bool TrainIsNeeded = false;

    cv::ocl::setUseOpenCL(false);
    MnistToMat("G:\\Lib\\caffeOLD\\data\\mnist\\train-images-idx3-ubyte", "G:\\Lib\\caffeOLD\\data\\mnist\\train-labels-idx1-ubyte",trainImages, trainLabels);
    MnistToMat("G:\\Lib\\caffeOLD\\data\\mnist\\t10k-images-idx3-ubyte", "G:\\Lib\\caffeOLD\\data\\mnist\\t10k-labels-idx1-ubyte", testImages, testLabels);
    if (trainImages.size() == 0 || trainImages.size() != trainLabels.size() || testImages.size() == 0 || testImages.size() != testLabels.size())
    {
        cout << "Error reading data or test file\n";
        return 0;
    }
    int nbItems = trainImages.size();
    int nbTests = testImages.size();

    vector<float> data, dataTest;
    vector<float> label, labelTest;

    VectorMat2VectorFloat(trainImages, trainLabels,data,label);
    VectorMat2VectorFloat(testImages, testLabels, dataTest, labelTest);
    if (TrainIsNeeded)
    {
        caffe::SolverParameter solver_param;
        caffe::ReadSolverParamsFromTextFileOrDie("lenet_solver.prototxt", &solver_param);

        boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
        caffe::MemoryDataLayer<float> *dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("data").get());
        caffe::MemoryDataLayer<float> *dataLayer_testnet_ = (caffe::MemoryDataLayer<float> *) (solver->test_nets()[0]->layer_by_name("test_inputdata").get());


        dataLayer_testnet_->Reset(dataTest.data(), labelTest.data(), testImages.size());

        dataLayer_trainnet->Reset(data.data(), label.data(), trainImages.size());

        solver->Solve();

        caffe::NetParameter net_param;
        solver->net()->ToProto(&net_param);
        caffe::WriteProtoToBinaryFile(net_param, "model.binary");
        caffe::WriteProtoToTextFile(net_param, "model.txt");
        


        boost::shared_ptr<caffe::Net<float> > testnet;
        testnet.reset(new caffe::Net<float>("lenet_train_test.prototxt", caffe::TEST));

        testnet->ShareTrainedLayersWith(solver->net().get());
        caffe::MemoryDataLayer<float> *dataLayer_testnet = (caffe::MemoryDataLayer<float> *) (testnet->layer_by_name("test_inputdata").get());
        dataLayer_testnet->Reset(dataTest.data(), labelTest.data(), 50);
        testnet->Forward();
        boost::shared_ptr<caffe::Blob<float> > output_layer = testnet->blob_by_name("ip2");
        const float* begin = output_layer->cpu_data();
        for (int i = 0; i < 50; i++)
        {
            cout << i <<  " <_> " << testLabels[i]<<" ";
            for (int j=0;j<10;j++)
                cout << begin[i*10+j]<<" " ;
            cout << endl;
        }
    
    }


    String modelTxt("lenet.prototxt");
    String modelBin("model.binary");

    dnn::Net net;
    try {
        net = dnn::readNetFromCaffe(modelTxt, modelBin);
    }
    catch (const cv::Exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        if (net.empty())
        {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "prototxt:   " << modelTxt << std::endl;
            std::cerr << "caffemodel: " << modelBin << std::endl;
            std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
            std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
            exit(-1);
        }
    }
    Mat img = testImages[0];
    
    Mat inputBlob = dnn::blobFromImage(img, 1 / 256.0,Size(),-0.1);
    net.setInput(inputBlob, "data");        //set the network input
    Mat prob = net.forward();         //compute output
    cout << prob << "\n";
    return 0;
}

void VectorMat2VectorFloat(vector<Mat> img, vector<uint> lab, vector<float> &data, vector<float> &label) 
{
    data.resize(img.size()*img[0].total());
    label.resize(img.size());


    for (int i = 0; i<img.size(); ++i)
    {
        Mat imF;
        img[i].convertTo(imF, CV_32F,1./256,-0.1);
        memcpy(&data[i*img[0].total()], imF.data, imF.total()*4);
        label[i] = lab[i];
    }

}


void MnistToMat(char *nameData,char *nameLabel,vector<Mat> &trainImages, std::vector<uint> &labels, bool display)
{
    ifstream fMnistTrain, fMnistLabel;

    fMnistTrain.open(nameData, ios::binary + ios::in);
    if (!fMnistTrain.is_open())
    {
        cout << "File not found\n";
        return ;
    }
    fMnistLabel.open(nameLabel, ios::binary + ios::in);
    if (!fMnistLabel.is_open())
    {
        cout << "File not found\n";
        return ;
    }

    int magicNumber, nbItems, nbRows, nbCols;
    fMnistTrain.read((char*)&magicNumber, 4);
    fMnistTrain.read((char*)&nbItems, 4);
    fMnistTrain.read((char*)&nbRows, 4);
    fMnistTrain.read((char*)&nbCols, 4);
    Lsb2Msb(nbItems);
    Lsb2Msb(nbRows);
    Lsb2Msb(nbCols);
    std::vector<cv::Mat> ;
    for (int i = 0; i < nbItems; i++)
    {
        cv::Mat x(nbRows, nbCols, CV_8UC1);
        fMnistTrain.read((char*)x.data, nbRows* nbCols);
        trainImages.push_back(x);
    }
    int nbLabels;
    fMnistLabel.read((char*)&magicNumber, 4);
    fMnistLabel.read((char*)&nbLabels, 4);
    Lsb2Msb(nbLabels);

    for (int i = 0; i < nbItems; i++)
    {
        uchar c;
        fMnistLabel.read((char*)&c, 1);
        labels.push_back(c);
        if (i % 1000 == 0 && display)
        {
            imshow(format("Img %d", c), trainImages[i]);
            waitKey(10);
        }
    }

}