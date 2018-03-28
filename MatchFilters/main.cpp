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

void main(void)
{
    vector<String> fileNames;
    // Structure N filtres par itération , M itération par fichiers, P réseaux filters[p][m][n]
    vector<vector<vector<Mat>>> filters;

    glob("conv2o*.yml", fileNames);
    int nbIterRef = -1;
    int nbFilters = -1;
    for (auto file : fileNames)
    {
        cout << "File : " << file << endl;
        FileStorage fs;
        fs.open(file, FileStorage::READ);
        
        int idxStream = 0;
        FileNode n;
        vector<vector<Mat>> w;
        do
        {
            n= fs.root(idxStream++);
            if (!n.empty())
            {
                cout << n.name() <<" "<< n.type()<< endl;
                FileNodeIterator it = n.begin();
                vector<Mat> x;
                for (; it != n.end(); it++)
                {
                    cout << (*it).name() << "\n";
                    fs[(*it).name()]>>x;
                }
                w.push_back(x);
            }
        } 
        while (!n.empty());
        filters.push_back(w);

    }
    for (int ra = 0; ra < filters.size(); ra++)
    {

        for (int itea = filters[ra].size()-4; itea < filters[ra].size(); itea++)
        {
            for (int idxFiltera = 0; idxFiltera < filters[ra][itea].size(); idxFiltera++)
            {
                cout << "File " << fileNames[ra] << " ** Ite " << itea << "** Filter " << idxFiltera << "\n";
                double x = mean(filters[ra][itea][idxFiltera])[0];
                for (int rb = 0; rb < filters.size(); rb++)
                {
                    int idxFile = -1;
                    for (int iteb = filters[rb].size()-4; iteb < filters[rb].size(); iteb++)
                    {
                        double m = -1;
                        int idxIte = -1;
                        for (int idxFilterb = 0; idxFilterb < filters[rb][iteb].size(); idxFilterb++)
                        {
                            double y = mean(filters[rb][iteb][idxFilterb])[0];
                            Mat p;
                            multiply(filters[ra][itea][idxFiltera] - x, filters[rb][iteb][idxFilterb] - y, p);
                            y = sum(p)[0];
                            if (y > m)
                            {
                                m = y;
                                idxIte = idxFilterb;
                            }

                        }
                        cout << idxIte << "\t";
                        if (idxFile == -1)
                            idxFile = idxIte;
                        else
                            if (idxFile != idxIte)
                                idxFile = -2;
                    }
                    cout << "----> " << idxFile << "(" << fileNames[rb] << ")";
                    if (idxFile == -2)
                        cout << "++++++++++++++++++++++\n";
                    else
                        cout << "\n";
                }
                cout << "\n**************************************************************\n";
            }
        }
        cout << "\n";
    }

}
