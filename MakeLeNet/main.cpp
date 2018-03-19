#include <caffe/caffe.hpp>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <fstream>

#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

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

void MnistToMat(char *nameData, char *nameLabel, vector<Mat> &trainImages, std::vector<uint> &labels,int nb, bool display = false);
void VectorMat2VectorFloat(vector<Mat> img, vector<uint> lab, vector<float> &data, vector<float> &label);

const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{solver or model prototxt s    |lenet_solverleger10.prototxt     | solver param }"
"{train model t     |1     | train model }"
"{ m model name    |     | model name }"
"{ b binary name    |     | binary name }"
"{ c binary name    |  10   | number of class }"
;


using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;
int main2() {
    caffe::NetParameter param;
    caffe::NetParameter param2;

    caffe::LayerParameter lparam;
    caffe::LayerParameter *lparam2;
    const char * filename = "lenet_train_Leger10.prototxt";
    fstream fd(filename, ios::binary|ios::in);
    google::protobuf::io::IstreamInputStream input1(&fd);
    if (!fd.is_open())
        cout << "File not found: " << filename;
    bool success = google::protobuf::TextFormat::Parse(&input1, &param);
    cout << "Network Name: " << param.name() << endl;
    if (param.input_size()!=0)
        cout << "Input: " << param.input(0) << endl;
    for (int j = 0; j < param.input_dim_size(); j++) {
        cout << "Input Dim " << j << ": " << param.input_dim(j) << endl;
    }
    cout << "Number of Layers (in implementation): " << param.layer_size() << endl << endl;
    for (int nlayers = 0; nlayers < param.layer_size(); nlayers++) {
        lparam = param.layer(nlayers);
        

        cout << endl << "Parameters for Layer " << nlayers + 1 << ":" << endl;
        cout << "Name: " << lparam.name() << endl;
        cout << "Type: " << lparam.type() << endl;
        for (int num_bottom_layers = 0; num_bottom_layers < lparam.bottom_size(); num_bottom_layers++) 
        {
            cout << "Bottom: " << lparam.bottom(num_bottom_layers) << endl;
        }
        for (int num_top_layers = 0; num_top_layers < lparam.top_size(); num_top_layers++) 
        {
            cout << "Top: " << lparam.top(num_top_layers) << endl;
        }
        for (int i = 0; i < lparam.param_size(); i++) 
        {
            cout << "LR_MULT: " << lparam.param(i).lr_mult() << endl;
            cout << "decay_MULT: " << lparam.param(i).decay_mult() << endl;
        }
        if (lparam.has_convolution_param()) 
        {
            boost::shared_ptr<caffe::ConvolutionParameter> x(new caffe::ConvolutionParameter(lparam.convolution_param()));
            for (int kSize = 0; kSize < lparam.convolution_param().pad().size(); kSize++)
            {
                cout << "Pad: " << lparam.convolution_param().pad(kSize) << endl;
            }
            for (int kSize = 0; kSize < lparam.convolution_param().stride().size(); kSize++)
            {
                cout << "Stride: " << lparam.convolution_param().stride(kSize) << endl;
            }
            cout << "Kernel Size: " << lparam.convolution_param().kernel_size().size() << endl;
            cout << "Number of Outputs: " << lparam.convolution_param().num_output() << endl;
            cout << "Group: " << lparam.convolution_param().group() << endl;
        }
        if (lparam.has_lrn_param()) 
        {
            cout << "Local Size: " << lparam.lrn_param().local_size() << endl;
            cout << "Alpha: " << lparam.lrn_param().alpha() << endl;
            cout << "Beta: " << lparam.lrn_param().beta() << endl;
        }
        if (lparam.has_pooling_param()) 
        {
            cout << "Pool: " << lparam.pooling_param().pool() << endl;
            cout << "Kernel Size: " << lparam.pooling_param().kernel_size() << endl;
            cout << "Stride: " << lparam.pooling_param().stride() << endl;
        }
        if (lparam.has_inner_product_param()) 
        {
            cout << "Number of Outputs: " << lparam.inner_product_param().num_output() << endl;
            if (nlayers == 2)
            {
                google::protobuf::uint32 x = 12;
                lparam.convolution_param(); 

            }
        }
        if (lparam.has_dropout_param()) 
        {
            cout << "Dropout Ratio: " << lparam.dropout_param().dropout_ratio() << endl;
        }
    }
    //close(fd);
    param2.set_name("toto.prototxt");

    return 0;
}

caffe::NetParameter CopyNet(string nomFichier) 
{
    caffe::NetParameter param;
    caffe::NetParameter param2;

    fstream fd(nomFichier.c_str(), ios::binary | ios::in);
    google::protobuf::io::IstreamInputStream input1(&fd);
    if (!fd.is_open())
        cout << "File not found: " << nomFichier;
    bool success = google::protobuf::TextFormat::Parse(&input1, &param);
    for (int kSize = 0; kSize < param.input_size(); kSize++)
    {
        *param2.add_input() = param.input(kSize);
    }
    for (int j = 0; j < param.input_dim_size(); j++) 
    {
        param2.add_input_dim(param.input_dim(j));
    }
    param2.set_name(param.name());
    for (int nlayers = 0; nlayers < param.layer_size(); nlayers++) 
    {
        caffe::LayerParameter *couche = new caffe::LayerParameter(param.layer(nlayers));
        *param2.add_layer() = *couche;
    }
    string s;
    return param2;
}

template <typename T>
caffe::NetParameter ModifyNet(string nomFichier,T &y,string layerName,int numOutput)
{
    caffe::NetParameter param;
    caffe::NetParameter param2;

    fstream fd(nomFichier.c_str(), ios::binary | ios::in);
    google::protobuf::io::IstreamInputStream input1(&fd);
    if (!fd.is_open())
        cout << "File not found: " << nomFichier;
    bool success = google::protobuf::TextFormat::Parse(&input1, &param);
    for (int kSize = 0; kSize < param.input_size(); kSize++)
    {
        *param2.add_input() = param.input(kSize);
    }
    for (int j = 0; j < param.input_dim_size(); j++)
    {
        param2.add_input_dim(param.input_dim(j));
    }
    param2.set_name(param.name());
    for (int nlayers = 0; nlayers < param.layer_size(); nlayers++)
    {
        caffe::LayerParameter *couche = new caffe::LayerParameter(param.layer(nlayers));
        if (couche->name() == layerName)
        {
            couche->mutable_convolution_param()->set_num_output(numOutput);
        }
        *param2.add_layer() = *couche;
    }
    string s;
    return param2;
}


int main(int argc, char **argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    caffe::Caffe::set_mode(caffe::Caffe::CPU);

    vector<Mat> trainImages;
    vector<uint> trainLabels;
    vector<Mat> testImages;
    vector<uint> testLabels;
    int nbLabels=10;
    bool TrainIsNeeded = true;

    MnistToMat("G:\\Lib\\caffeOLD\\data\\mnist\\train-images-idx3-ubyte", "G:\\Lib\\caffeOLD\\data\\mnist\\train-labels-idx1-ubyte", trainImages, trainLabels, nbLabels);
    MnistToMat("G:\\Lib\\caffeOLD\\data\\mnist\\t10k-images-idx3-ubyte", "G:\\Lib\\caffeOLD\\data\\mnist\\t10k-labels-idx1-ubyte", testImages, testLabels, nbLabels);
    if (trainImages.size() == 0 || trainImages.size() != trainLabels.size() || testImages.size() == 0 || testImages.size() != testLabels.size())
    {
        cout << "Error reading data or test file\n";
        return 0;
    }
    int nbItems = trainImages.size();
    int nbTests = testImages.size();
    trainImages.resize((nbItems / 50) * 50);
    trainLabels.resize((nbItems / 50) * 50);
    testImages.resize((nbTests / 50) * 50);
    testLabels.resize((nbTests / 50) * 50);
    nbItems = trainImages.size();
    nbTests = testImages.size();

    vector<float> data, dataTest;
    vector<float> label, labelTest;

    VectorMat2VectorFloat(trainImages, trainLabels, data, label);
    VectorMat2VectorFloat(testImages, testLabels, dataTest, labelTest);


    for (int numOutput = 1; numOutput < 10; numOutput++)
    {

        caffe::ConvolutionParameter x;
        caffe::NetParameter net2 = ModifyNet("lenet_train_Leger.prototxt", x, "conv1", numOutput);


        string nomFichier(format("lenet_conv1_%d.prototxt", numOutput)),s;
        fstream f(nomFichier.c_str(), ios::out);
        google::protobuf::TextFormat::PrintToString(net2, &s);
        f.write(s.c_str(), s.length());
        f.flush();
        f.close();

    

        caffe::SolverParameter solver_param1,solver_param;

        caffe::ReadSolverParamsFromTextFileOrDie("lenet_solver.prototxt", &solver_param1);
        (*solver_param.mutable_net_param()) = net2;
        *solver_param.mutable_lr_policy() = "inv";
//        *solver_param.mutable_net() = nomFichier;
        solver_param.add_test_iter(1);
        solver_param.set_test_iter(0, 200);
        solver_param.set_test_interval(100);
        solver_param.set_base_lr(0.01);
        solver_param.set_momentum(0.9);
        solver_param.set_gamma(0.0001);
        solver_param.set_power(0.75);
        solver_param.set_max_iter(300);



        boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));


        caffe::MemoryDataLayer<float> *dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("data").get());
        caffe::MemoryDataLayer<float> *dataLayer_testnet_ = (caffe::MemoryDataLayer<float> *) (solver->test_nets()[0]->layer_by_name("test_inputdata").get());

        dataLayer_testnet_->Reset(dataTest.data(), labelTest.data(), testImages.size());

        dataLayer_trainnet->Reset(data.data(), label.data(), trainImages.size());

        solver->Solve();

        caffe::NetParameter net_param;
        solver->net()->ToProto(&net_param);
        caffe::WriteProtoToBinaryFile(net_param, nomFichier +".binary");
        caffe::WriteProtoToTextFile(net_param, nomFichier +".txt");
        


        boost::shared_ptr<caffe::Net<float> > testnet;
        testnet.reset(new caffe::Net<float>(nomFichier, caffe::TEST));

        testnet->ShareTrainedLayersWith(solver->net().get());
        caffe::MemoryDataLayer<float> *dataLayer_testnet = (caffe::MemoryDataLayer<float> *) (testnet->layer_by_name("test_inputdata").get());
        dataLayer_testnet->Reset(dataTest.data(), labelTest.data(), 50);
        testnet->Forward();
        boost::shared_ptr<caffe::Blob<float> > output_layer = testnet->blob_by_name("ip2");
        const float* begin = output_layer->cpu_data();
        for (int i = 0; i < 50; i++)
        {
            cout << i <<  " <_> " << testLabels[i]<<" ";
            for (int j=0;j<nbLabels;j++)
                cout << begin[i*nbLabels +j]<<" " ;
            cout << endl;
        }
    
    }


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


void MnistToMat(char *nameData,char *nameLabel,vector<Mat> &trainImages, std::vector<uint> &labels,int nb, bool display)
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
    int nbLabels;
    fMnistLabel.read((char*)&magicNumber, 4);
    fMnistLabel.read((char*)&nbLabels, 4);
    Lsb2Msb(nbLabels);
    for (int i = 0; i < nbItems; i++)
    {
        uchar c;
        cv::Mat x(nbRows, nbCols, CV_8UC1);
        fMnistLabel.read((char*)&c, 1);
        fMnistTrain.read((char*)x.data, nbRows* nbCols);
        if (c < nb)
        {
            labels.push_back(c);
            trainImages.push_back(x);

        }
        if (i % 1000 == 0 && display)
        {
            imshow(format("Img %d", c), trainImages[i]);
            waitKey(10);
        }
    }

}