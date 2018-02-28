#include <caffe/caffe.hpp>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
#include <fstream>
#include <numeric>
#include "caffe/layers/memory_data_layer.hpp"

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
struct BaseFormes {
//    Ptr<TrainData> baseDonnees;
//    Ptr<TrainData> baseDonneesANN;
    vector<vector<Point>> formes;
    int nbFormeParNiveauBruit = 40;
    int nbNiveauBruit = 8;
    vector<float> minF;
    vector<float> maxF;
};
/* Find best class for the blob (i. e. class with maximal probability) */
static int getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
    Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
    Point classNumber;
    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    return  classNumber.x;
}

struct OutilsDessin {
    Mat feuille;
    int tailleCrayon;
    int couleurCrayon;
    bool effaceFeuille = true;
};

void DonneesModele(BaseFormes &,  vector<Mat> &, vector<uint> &,int);
vector<Point> NoisyPolygon(vector<Point> pRef, double n);
void GestionCrayon(int evt, int x, int y, int type, void *extra);
void NormaliseContour(vector<Point> &c, int tailleBoite);

void MnistToMat(char *nameData, char *nameLabel, vector<Mat> &trainImages, std::vector<uint> &labels, bool display = false);
void VectorMat2VectorFloat(vector<Mat> img, vector<uint> lab, vector<float> &data, vector<float> &label);

const String keys =
"{Aide h usage ? help  |     | Afficher ce message   }"
"{solver or model prototxt s    |lenet_solvershape.prototxt     | solver param }"
"{train model t     |0     | train model }"
"{ m model name    | lenet_Shape.prototxt    | model name }"
"{ b binary name    |lenet_train_Shape.binary     | binary name }"
;



int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    String solverParamName = parser.get<String>("s");
    bool trainIsNeeded;
    if (parser.get<int>("t"))
        trainIsNeeded = true;
    else
        trainIsNeeded = false;


    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    vector<Mat> trainImages;
    vector<uint> trainLabels;
    vector<Mat> testImages;
    vector<uint> testLabels;
    vector<Mat> exemple;

    cv::ocl::setUseOpenCL(false);
    MnistToMat("G:\\Lib\\caffeOLD\\data\\mnist\\train-images-idx3-ubyte", "G:\\Lib\\caffeOLD\\data\\mnist\\train-labels-idx1-ubyte", trainImages, trainLabels);
    MnistToMat("G:\\Lib\\caffeOLD\\data\\mnist\\t10k-images-idx3-ubyte", "G:\\Lib\\caffeOLD\\data\\mnist\\t10k-labels-idx1-ubyte", testImages, testLabels);
    BaseFormes md;
    if (!trainIsNeeded)
    {
        md.nbFormeParNiveauBruit = 1;
        md.nbNiveauBruit = 1;
    }
    DonneesModele(md, trainImages, trainLabels, 10);
    DonneesModele(md, testImages, testLabels, 10);
    if (trainImages.size() == 0 || trainImages.size() != trainLabels.size() || testImages.size() == 0 || testImages.size() != testLabels.size())
    {
        cout << "Error reading data or test file\n";
        return 0;
    }
    int nbItems = trainImages.size();
    int nbTests = testImages.size();

    vector<float> data, dataTest;
    vector<float> label, labelTest;

    VectorMat2VectorFloat(trainImages, trainLabels, data, label);
    VectorMat2VectorFloat(testImages, testLabels, dataTest, labelTest);
    caffe::SolverParameter solver_param;
    if (trainIsNeeded)
    {
        caffe::ReadSolverParamsFromTextFileOrDie(solverParamName, &solver_param);
        boost::shared_ptr<caffe::Solver<float> > solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
        caffe::MemoryDataLayer<float> *dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (solver->net()->layer_by_name("data").get());
        caffe::MemoryDataLayer<float> *dataLayer_testnet_ = (caffe::MemoryDataLayer<float> *) (solver->test_nets()[0]->layer_by_name("test_inputdata").get());


        dataLayer_testnet_->Reset(dataTest.data(), labelTest.data(), testImages.size());

        dataLayer_trainnet->Reset(data.data(), label.data(), trainImages.size());

        solver->Solve();

        caffe::NetParameter net_param;
        solver->net()->ToProto(&net_param);
        caffe::WriteProtoToBinaryFile(net_param, solver_param.net() + ".binary");
        caffe::WriteProtoToTextFile(net_param, solver_param.net() + ".txt");



        boost::shared_ptr<caffe::Net<float> > testnet;
        testnet.reset(new caffe::Net<float>(solver_param.net(), caffe::TEST));

        testnet->ShareTrainedLayersWith(solver->net().get());
        caffe::MemoryDataLayer<float> *dataLayer_testnet = (caffe::MemoryDataLayer<float> *) (testnet->layer_by_name("test_inputdata").get());
        dataLayer_testnet->Reset(dataTest.data(), labelTest.data(), 50);
        testnet->Forward();
        boost::shared_ptr<caffe::Blob<float> > output_layer = testnet->blob_by_name("ip2");
        const float* begin = output_layer->cpu_data();
        for (int i = 0; i < 50; i++)
        {
            cout << i << " <_> " << labelTest[i] ;
            for (int j = 0; j<15; j++)
                cout << begin[i * 15 + j] << " ";
            cout << endl;
        }

    }


    
    String modelTxt;// ("lenetleger.prototxt");
    String modelBin;// ("lenetleger.binary");
    if (trainIsNeeded)
    {
        modelTxt = solver_param.net() + ".prototxt";
        modelBin = solver_param.net() + ".binary";
    }
    else
    {
        modelTxt = parser.get<String>("m");;
        modelBin = parser.get<String>("b");;
    }

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
            exit(-1);
        }
    }
    int error = 0;
    int ind = 0;




    int labelRef = 0;
    for (auto &item : testImages)
    {
        if (labelRef == testLabels[ind])
        {
            exemple.push_back(item);
            labelRef++;

        }
        if (testLabels[ind] >= 10)
        {
            Mat img = item;
            Mat inputBlob = dnn::blobFromImage(img, 1 / 256.0, Size(), -0.1);
            imshow("img", img);
            net.setInput(inputBlob, "data");        //set the network input
            Mat prob = net.forward();         //compute output
            int classId;
            double p;
        
            cout << testLabels[ind]<< " " << getMaxClass(prob, &classId, &p)<<" ";
            cout << prob << "\n";

        }
        ind++;

    }
    Mat alphabet(280, 280, CV_8UC1, Scalar(0));

    ind = 0;
    for (auto &item : exemple)
    {
        item.copyTo(alphabet(Rect((ind%10) * 28, ind / 10 * 28, 28, 28)));
        ind++;
    }
    imshow("Alphabet", alphabet);
    String nomFenetre = "Feuille";
    namedWindow(nomFenetre);
    OutilsDessin abcdere;

    abcdere.feuille = Mat::zeros(16*28, 16*28, CV_8UC1);
    abcdere.couleurCrayon = 255;
    abcdere.tailleCrayon = 4;
    setMouseCallback(nomFenetre, GestionCrayon, &abcdere);
    int code = 0;
    do
    {
        imshow("Feuille", abcdere.feuille);
        code = waitKey(10);
        if (code >= '0' && code<'0' + md.formes.size())
        {
            abcdere.feuille.setTo(0);
            if (code - 48<md.formes.size())
                drawContours(abcdere.feuille, md.formes, code - 48, Scalar(abcdere.couleurCrayon), abcdere.tailleCrayon);
        }
        switch (code) {
        case 'e':
            abcdere.feuille.setTo(0);
            destroyWindow("formes retenues");
            destroyWindow("Classement");
            break;
        case '+':
            abcdere.tailleCrayon++;
            if (abcdere.tailleCrayon>10)
                abcdere.tailleCrayon = 10;
            break;
        case '-':
            abcdere.tailleCrayon--;
            if (abcdere.tailleCrayon<1)
                abcdere.tailleCrayon = 1;
            break;
        case 'c':
        {
            Mat img;
            resize(abcdere.feuille, img, Size(28, 28));
            imshow("img", img);
            Mat inputBlob = dnn::blobFromImage(img, 1 / 256.0, Size(28,28), -0.1);
            net.setInput(inputBlob, "data");        //set the network input
            Mat prob = net.forward();         //compute output
            int classId;
            double p;

            cout <<  "\n"<<getMaxClass(prob, &classId, &p) << " ";
            cout << prob << "\n";
            waitKey();
        }

            break;
         }

    } while (code != 27);

    return 0;
}

void VectorMat2VectorFloat(vector<Mat> img, vector<uint> lab, vector<float> &data, vector<float> &label)
{
    vector<int>  a(img.size());
    std::iota(std::begin(a), std::end(a), 0);
    random_shuffle(std::begin(a), std::end(a));

    data.resize(img.size()*img[0].total());
    label.resize(img.size());


    for (int i = 0; i<img.size(); ++i)
    {
        Mat imF;
        int ind = a[i];
        img[ind].convertTo(imF, CV_32F, 1. / 256, -0.1);
        memcpy(&data[i*img[0].total()], imF.data, imF.total() * 4);
        label[i] = lab[ind];
    }

}


void MnistToMat(char *nameData, char *nameLabel, vector<Mat> &trainImages, std::vector<uint> &labels, bool display)
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
    nbItems = nbItems / 10;
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

vector<Point> NoisyPolygon(vector<Point> pRef, double n)
{
    RNG rng;
    vector<Point> c;
    vector<Point> p = pRef;
    vector<vector<Point>> contour;
    for (int i = 0; i<p.size(); i++)
        p[i] += Point(static_cast<int>(n*rng.uniform(-1.0, 1.0)), static_cast<int>(n*rng.uniform(-1.0, 1.0)));
    c.push_back(p[0]);
    int minX = p[0].x, maxX = p[0].x, minY = p[0].y, maxY = p[0].y;
    for (int i = 0; i <p.size(); i++)
    {
        int next = i + 1;
        if (next == p.size())
            next = 0;
        Point2d u = p[next] - p[i];
        int d = static_cast<int>(norm(u));
        double a = atan2(u.y, u.x);
        int step = 1;
        if (n != 0)
            step = static_cast<int>(d / n);
        for (int j = 1; j<d; j += max(step, 1))
        {
            Point pNew;
            do
            {
                Point2d pAct = (u*j) / static_cast<double>(d);
                double r = n * rng.uniform((double)0, (double)1);
                double theta = a + rng.uniform(0., 2 * CV_PI);
                pNew = Point(static_cast<int>(r*cos(theta) + pAct.x + p[i].x), static_cast<int>(r*sin(theta) + pAct.y + p[i].y));
            } while (pNew.x<0 || pNew.y<0);
            if (pNew.x<minX)
                minX = pNew.x;
            if (pNew.x>maxX)
                maxX = pNew.x;
            if (pNew.y<minY)
                minY = pNew.y;
            if (pNew.y>maxY)
                maxY = pNew.y;
            c.push_back(pNew);
        }
    }
    contour.push_back(c);
    Mat frame(maxY + 2, maxX + 2, CV_8UC1, Scalar(0));
    drawContours(frame, contour, 0, Scalar(255), -1);
    findContours(frame, contour, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    return contour[0];
}

void DonneesModele(BaseFormes &md,vector<Mat> &t,vector<uint> &l,int ol)
{
    FileStorage forme("formes.yml", FileStorage::READ);
    if (!forme.isOpened() || forme["PolygoneRef"].empty())
    {
        forme.release();
        vector<Point>  vertex;

        vertex.push_back(Point(300, 300));
        vertex.push_back(Point(500, 300));
        vertex.push_back(Point(500, 500));
        md.formes.push_back(vertex);
        vertex.push_back(Point(300, 500));
        md.formes.push_back(vertex);
        vertex.push_back(Point(200, 600));
        md.formes.push_back(vertex);
        vertex.push_back(Point(100, 100));
        md.formes.push_back(vertex);
        FileStorage f("formes.yml", FileStorage::WRITE);
        f << "PolygoneRef" << md.formes;
    }
    else
    {
        forme["PolygoneRef"] >> md.formes;
        forme.release();
    }
    if (md.nbFormeParNiveauBruit == 0)
        return;
    int nbLigne = static_cast<int>(md.formes.size())*md.nbFormeParNiveauBruit*md.nbNiveauBruit;
    ofstream fs;
    fs.open("ImageLabel.txt", ios::app);
    for (int indForme = 0; indForme < md.formes.size(); indForme++)
    {
        int offsetRow = indForme * md.nbFormeParNiveauBruit*md.nbNiveauBruit;
        double noiseLevel = 3;
        for (int i = 0; i < md.nbNiveauBruit; i++)
        {
            for (int j = 0; j < md.nbFormeParNiveauBruit; j++)
            {
                vector<Point> c = NoisyPolygon(md.formes[indForme], noiseLevel);
                Mat frame(64, 64, CV_8UC1, Scalar(0));
                vector<vector<Point>> contour;
                NormaliseContour(c, 100);
                contour.push_back(c);
                drawContours(frame, contour, static_cast<int>(contour.size()) - 1, Scalar(255), 2);
                imshow("Contour Ref.", frame);
                resize(frame, frame, Size(28, 28));
                t.push_back(frame);
                l.push_back(indForme + ol);
                waitKey(1);
            }
            noiseLevel += 1.5;
        }
    }
    destroyWindow("Contour Ref.");
}

void GestionCrayon(int evt, int x, int y, int type, void *extra)
{
    OutilsDessin *pgc = (OutilsDessin*)extra;
    if (type == EVENT_FLAG_LBUTTON)
    {
        if (pgc->effaceFeuille)
        {
            pgc->feuille.setTo(0);
            pgc->effaceFeuille = false;
        }
        if (x < pgc->feuille.cols && y < pgc->feuille.cols)
            circle(pgc->feuille, Point(x, y), pgc->tailleCrayon, Scalar(pgc->couleurCrayon), -1);
    }
}

void NormaliseContour(vector<Point> &c, int tailleBoite)
{
    Rect r = boundingRect(c);
    double k = 1.0 / max(r.br().x - r.tl().x, r.br().y - r.tl().y)*(tailleBoite - 50);
    for (int i = 0; i < c.size(); i++)
    {
        c[i] = (c[i] - r.tl())*k + Point(5, 5);
    }
}

