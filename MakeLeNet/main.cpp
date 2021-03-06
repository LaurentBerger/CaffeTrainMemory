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
    for (int i = 0; i<nb / 2; i++)
        swap(x[i], x[nb - i - 1]);
}

void MnistToMat(char *nameData, char *nameLabel, vector<Mat> &trainImages, std::vector<uint> &labels, int nb, bool display = false);
void VectorMat2VectorFloat(vector<Mat> img, vector<uint> lab, vector<float> &data, vector<float> &label);
void SaveWeightLayer(boost::shared_ptr<caffe::Net<float> > &reseau, string &searchName, string &fileName, int iter);

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
    fstream fd(filename, ios::binary | ios::in);
    google::protobuf::io::IstreamInputStream input1(&fd);
    if (!fd.is_open())
        cout << "File not found: " << filename;
    bool success = google::protobuf::TextFormat::Parse(&input1, &param);
    cout << "Network Name: " << param.name() << endl;
    if (param.input_size() != 0)
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
caffe::NetParameter ModifyNet(string nomFichier, T &y, string layerName, int numOutput)
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
            //            couche->mutable_convolution_param()->set_num_output(numOutput);
            couche->mutable_inner_product_param()->set_num_output(numOutput);
        }
        if (couche->name() == "conv1")
        {
            couche->mutable_convolution_param()->set_num_output(4);
        }
        *param2.add_layer() = *couche;
    }
    string s;
    return param2;
}


int main(int argc, char **argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    FLAGS_alsologtostderr = 0;
    ::google::InitGoogleLogging(argv[0]);
    caffe::Caffe::set_mode(caffe::Caffe::CPU);

    int nbLabels = 10;
    bool TrainIsNeeded = true;




    for (int numOutput = 4; numOutput <= 10; numOutput++)
    {
        vector<Mat> trainImages;
        vector<uint> trainLabels;
        vector<Mat> testImages;
        vector<uint> testLabels;
        nbLabels = numOutput;
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

        caffe::ConvolutionParameter x;
        caffe::NetParameter net2 = ModifyNet("lenet_train_Leger.prototxt", x, "ip2", numOutput);


        string nomFichier(format("lenet_conv1_%d.prototxt", numOutput)), s;
        fstream f(nomFichier.c_str(), ios::out);
        google::protobuf::TextFormat::PrintToString(net2, &s);
        f.write(s.c_str(), s.length());
        f.flush();
        f.close();



        caffe::SolverParameter solver_param1, solver_param;

        caffe::ReadSolverParamsFromTextFileOrDie("lenet_solver.prototxt", &solver_param1);
        int iterStep = 4000;
        (*solver_param.mutable_net_param()) = net2;
        *solver_param.mutable_lr_policy() = "inv";
        //        *solver_param.mutable_net() = nomFichier;
        solver_param.add_test_iter(1);
        solver_param.set_test_iter(0, 200);
        solver_param.set_test_interval(iterStep);
        solver_param.set_base_lr(0.01);
        solver_param.set_momentum(0.9);
        solver_param.set_gamma(0.0001);
        solver_param.set_power(0.75);
        solver_param.set_max_iter(iterStep);



        boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));


        caffe::MemoryDataLayer<float> *dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("data").get());
        caffe::MemoryDataLayer<float> *dataLayer_testnet_ = (caffe::MemoryDataLayer<float> *) (solver->test_nets()[0]->layer_by_name("test_inputdata").get());

        dataLayer_testnet_->Reset(dataTest.data(), labelTest.data(), testImages.size());

        dataLayer_trainnet->Reset(data.data(), label.data(), trainImages.size());
        FLAGS_alsologtostderr = 1;

        solver->Solve();
        string layerName1("conv1");
        string fileName1(layerName1 + "o" + to_string(numOutput) + "_");
        string layerName2("conv2");
        string fileName2(layerName2 + "o" + to_string(numOutput) + "_");
        int nbFile = 0;
        SaveWeightLayer(solver->net(), layerName1, fileName1, nbFile);
        SaveWeightLayer(solver->net(), layerName2, fileName2, nbFile++);
        for (int iter = 1000; iter < 40000; iter += iterStep)
        {
            cout << "\n**********************************************************\n";
            solver->Step(iterStep);
            SaveWeightLayer(solver->net(), layerName1, fileName1, nbFile);
            SaveWeightLayer(solver->net(), layerName2, fileName2, nbFile++);

        }
        caffe::NetParameter net_param;
        solver->net()->ToProto(&net_param);
        string snap(nomFichier + ".binary");
        caffe::WriteProtoToBinaryFile(net_param, snap);

        FLAGS_alsologtostderr = 0;

        caffe::WriteProtoToTextFile(net_param, nomFichier + ".txt");


        {
            boost::shared_ptr<caffe::Net<float> > testnet;
            testnet.reset(new caffe::Net<float>(nomFichier, caffe::TEST));

            testnet->ShareTrainedLayersWith(solver->net().get());
            caffe::MemoryDataLayer<float> *dataLayer_testnet = (caffe::MemoryDataLayer<float> *) (testnet->layer_by_name("test_inputdata").get());
            dataLayer_testnet->Reset(dataTest.data(), labelTest.data(), 10000);
            testnet->Forward();
            boost::shared_ptr<caffe::Blob<float> > output_layer = testnet->blob_by_name("ip2");
            boost::shared_ptr<caffe::Blob<float> > loss_layer = testnet->blob_by_name("loss");
            boost::shared_ptr<caffe::Blob<float> > accuracy_layer = testnet->blob_by_name("accuracy");
            cout << "acc =" << *accuracy_layer->cpu_data() << "\t";
            cout << "loss = " << *loss_layer->cpu_data() << "\n";

        }
        {
            boost::shared_ptr<caffe::Net<float> > testnet;
            testnet.reset(new caffe::Net<float>(nomFichier, caffe::TEST));

            testnet->ShareTrainedLayersWith(solver->net().get());
            caffe::MemoryDataLayer<float> *dataLayer_testnet = (caffe::MemoryDataLayer<float> *) (testnet->layer_by_name("test_inputdata").get());
            dataLayer_testnet->Reset(dataTest.data(), labelTest.data(), 5000);
            testnet->Forward();
            boost::shared_ptr<caffe::Blob<float> > output_layer = testnet->blob_by_name("ip2");
            boost::shared_ptr<caffe::Blob<float> > loss_layer = testnet->blob_by_name("loss");
            boost::shared_ptr<caffe::Blob<float> > accuracy_layer = testnet->blob_by_name("accuracy");
            cout << "acc =" << *accuracy_layer->cpu_data() << "\t";
            cout << "loss = " << *loss_layer->cpu_data() << "\n";

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
        img[i].convertTo(imF, CV_32F, 1. / 256, -0.1);
        memcpy(&data[i*img[0].total()], imF.data, imF.total() * 4);
        label[i] = lab[i];
    }

}


void MnistToMat(char *nameData, char *nameLabel, vector<Mat> &trainImages, std::vector<uint> &labels, int nb, bool display)
{
    ifstream fMnistTrain, fMnistLabel;

    fMnistTrain.open(nameData, ios::binary + ios::in);
    if (!fMnistTrain.is_open())
    {
        cout << "File not found\n";
        return;
    }
    fMnistLabel.open(nameLabel, ios::binary + ios::in);
    if (!fMnistLabel.is_open())
    {
        cout << "File not found\n";
        return;
    }

    int magicNumber, nbItems, nbRows, nbCols;
    fMnistTrain.read((char*)&magicNumber, 4);
    fMnistTrain.read((char*)&nbItems, 4);
    fMnistTrain.read((char*)&nbRows, 4);
    fMnistTrain.read((char*)&nbCols, 4);
    Lsb2Msb(nbItems);
    Lsb2Msb(nbRows);
    Lsb2Msb(nbCols);
    std::vector<cv::Mat>;
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



void SaveWeightLayer(boost::shared_ptr<caffe::Net<float> > &reseau, string &searchName, string &fileName, int iter)
{
    vector<string> nomCouche;
    vector<string> nomBlob;

    nomCouche = reseau->layer_names();
    nomBlob = reseau->blob_names();
    FileStorage fs;
    fs.open("weights.yml", FileStorage::WRITE);
    vector<Mat> selectFilter;
    for (auto &nom : nomCouche)
    {
        const boost::shared_ptr<caffe::Layer<float> > &conv_layer = reseau->layer_by_name(nom);
        fs << nom << "[";
        fs << "{";
        fs << "Type" << conv_layer->type();
        if (conv_layer->blobs().size() > 0)
        {
            boost::shared_ptr<caffe::Blob<float> >& weight = conv_layer->blobs()[0];
            float* conv_weight = weight->mutable_cpu_data();
            if (fs.isOpened())
            {
                vector<int> s = weight->shape();
                fs << "WeightsShape" << s;
                switch (s.size())
                {
                case 1:
                {
                    fs << "M";
                    Mat a(s[0], 1, CV_32FC1, conv_weight);
                    fs << a;
                    break;

                }
                case 2:
                {
                    fs << "M";
                    Mat a(s[0], s[1], CV_32FC1, conv_weight);
                    fs << a;
                }
                break;
                case 3:
                    for (int i = 0; i < s[0]; i++)
                    {
                        fs << format("M%d", i);
                        Mat a(s[1], s[2], CV_32FC1, conv_weight + s[2] * s[3] * i);
                        fs << a;
                    }
                    break;
                case 4:

                    for (int i = 0; i < s[0]; i++)
                    {
                        for (int j = 0; j < s[1]; j++)
                        {
                            fs << format("M%dX%d", i, j);
                            Mat a(s[2], s[3], CV_32FC1, conv_weight + s[2] * s[3] * s[1] * i + s[2] * s[3] * j);
                            fs << a;
                            if (nom == searchName)
                                selectFilter.push_back(a.clone());
                        }
                    }
                    break;

                }

            }
            if (conv_layer->blobs().size() > 1)
            {
                boost::shared_ptr<caffe::Blob<float> >& bias = conv_layer->blobs()[1];
                float* conv_bias = bias->mutable_cpu_data();
                if (fs.isOpened())
                {
                    vector<int> s = bias->shape();
                    fs << "BiasShape" << s;
                    switch (s.size())
                    {
                    case 1:
                    {
                        fs << "B";
                        Mat a(s[0], 1, CV_32FC1, conv_bias);
                        fs << a;
                        break;

                    }
                    case 2:
                    {
                        fs << "B";
                        Mat a(s[0], s[1], CV_32FC1, conv_bias);
                        fs << a;
                    }
                    break;
                    case 3:
                        for (int i = 0; i < s[0]; i++)
                        {
                            fs << format("B%d", i);
                            Mat a(s[1], s[2], CV_32FC1, conv_bias + s[2] * s[3] * i);
                            fs << a;
                        }
                        break;
                    case 4:
                        for (int i = 0; i < s[0]; i++)
                        {
                            for (int j = 0; j < s[1]; j++)
                            {
                                fs << format("B%dX%d", i, j);
                                Mat a(s[2], s[3], CV_32FC1, conv_bias + s[2] * s[3] * s[1] * i + s[2] * s[3] * j);
                                fs << a;
                            }
                        }
                        break;

                    }

                }

            }

        }
        fs << "}";
        fs << "]";

    }

    if (selectFilter.size() != 0)
    {
        int ind = 0;
        FileStorage fs(format("%s.yml", fileName.c_str()), FileStorage::APPEND);
        string step("iter" + to_string(iter));
        fs << step ;
        fs << selectFilter;
        for (auto &img : selectFilter)
        {
            Mat x(256, 256, CV_32FC1, Scalar(0));
            img.copyTo(x(Rect(Point(0, 0), Size(img.cols, img.rows))));
            String nameIter = format("Image%d", ind++);
//            fs << nameIter << img;
        }
        fs.release();
    }
}