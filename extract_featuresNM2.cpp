#include "opencv2/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#define PI 3.14159265

int GroundTruth(cv::Mat& _originalImage)
{
    cv::Mat originalImage(_originalImage.rows + 2, _originalImage.cols + 2, _originalImage.type());
    copyMakeBorder(_originalImage, originalImage, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    cv::Mat bwImage(originalImage.size(), CV_8UC1);

    uchar thresholdValue = 100;
    uchar maxValue = 255;
    uchar middleValue = 192;
    uchar zeroValue = 0;
    cv::Scalar middleScalar(middleValue);
    cv::Scalar zeroScalar(zeroValue);

    static int neigborsCount = 4;
    static int dx[] = { -1,  0, 0, 1 };
    static int dy[] = { 0, -1, 1, 0 };
    int di, rx, ry;
    int perimeter;

    //cvtColor(originalImage, bwImage, CV_RGB2GRAY);
    threshold(originalImage, bwImage, thresholdValue, maxValue, cv::THRESH_BINARY_INV);

    int regionsCount = 0;
    int totalPixelCount = bwImage.rows * bwImage.cols;
    cv::Point seedPoint;
    cv::Rect rectFilled;
    int valuesSum, q1, q2, q3;
    bool p00, p10, p01, p11;

    for (int i = 0; i < totalPixelCount; i++)
    {
        if (bwImage.data[i] == maxValue)
        {
            seedPoint.x = i % bwImage.cols;
            seedPoint.y = i / bwImage.cols;

            if ((seedPoint.x == 0)
                || (seedPoint.y == 0)
                || (seedPoint.x == bwImage.cols - 1)
                || (seedPoint.y == bwImage.rows - 1))
            {
                continue;
            }

            regionsCount++;

            size_t pixelsFilled = floodFill(bwImage, seedPoint, middleScalar, &rectFilled);

            perimeter = 0;
            q1 = 0; q2 = 0; q3 = 0;

            int* crossings = new int[rectFilled.height];
            for (int j = 0; j < rectFilled.height; j++)
            {
                crossings[j] = 0;
            }

            for (ry = rectFilled.y - 1; ry <= rectFilled.y + rectFilled.height; ry++)
            {
                for (rx = rectFilled.x - 1; rx <= rectFilled.x + rectFilled.width; rx++)
                {
                    if ((bwImage.at<uint8_t>(ry, rx - 1) != bwImage.at<uint8_t>(ry, rx))
                        && (bwImage.at<uint8_t>(ry, rx - 1) + bwImage.at<uint8_t>(ry, rx) == middleValue + zeroValue))
                    {
                        crossings[ry - rectFilled.y]++;
                    }

                    if (bwImage.at<uint8_t>(ry, rx) == middleValue)
                    {
                        for (di = 0; di < neigborsCount; di++)
                        {
                            int xNew = rx + dx[di];
                            int yNew = ry + dy[di];
                            if (bwImage.at<uint8_t>(yNew, xNew) == zeroValue)
                            {
                                perimeter++;
                            }
                        }
                    }

                    p00 = bwImage.at<uint8_t>(ry, rx) == middleValue;
                    p01 = bwImage.at<uint8_t>(ry, rx + 1) == middleValue;
                    p10 = bwImage.at<uint8_t>(ry + 1, rx) == middleValue;
                    p11 = bwImage.at<uint8_t>(ry + 1, rx + 1) == middleValue;
                    valuesSum = p00 + p01 + p10 + p11;

                    if (valuesSum == 1)
                    {
                        q1++;
                    }
                    else if (valuesSum == 3)
                    {
                        q2++;
                    }
                    else if ((valuesSum == 2) && (p00 == p11))
                    {
                        q3++;
                    }
                }
            }

            q1 = q1 - q2 + 2 * q3;
            if (q1 % 4 != 0)
            {
                printf("Non-integer Euler number");
                exit(0);
            }
            q1 /= 4;

            std::vector<int> m_crossings;
            m_crossings.push_back(crossings[(int)rectFilled.height / 6]);
            m_crossings.push_back(crossings[(int)3 * rectFilled.height / 6]);
            m_crossings.push_back(crossings[(int)5 * rectFilled.height / 6]);
            sort(m_crossings.begin(), m_crossings.end());

            // Features used in the first stage classifier
            // aspect ratio (w/h), compactness (sqrt(a/p), number of holes (1 − η),
            // and a horizontal crossings feature (cˆ = median {c_1*w/6, c_3*w/6, c_5*w/6}) 
            // which estimates number of character strokes in horizontal projection
            if ((rectFilled.width >= 3) && (rectFilled.height >= 3)) // TODO find a better way to select good negative examples
            {
                printf("%f,%f,%f,%f,",
                    static_cast<float>(rectFilled.width) / rectFilled.height,
                    std::sqrt(pixelsFilled) / perimeter,
                    static_cast<float>(1 - q1),
                    static_cast<float>(m_crossings.at(1)));

                cv::Mat region = cv::Mat::zeros(bwImage.rows + 2, bwImage.cols + 2, CV_8UC1);
                int newMaskVal = 255;
                int flags = 4 + (newMaskVal << 8) + cv::FLOODFILL_FIXED_RANGE;
                cv::Rect rect;
                cv::floodFill(bwImage, region, seedPoint, zeroScalar, &rect, cv::Scalar(), cv::Scalar(), flags);
                rect.width += 2;
                rect.height += 2;
                region = region(rect);

                std::vector<std::vector<cv::Point> > contours;
                std::vector<cv::Point> contour_poly;
                std::vector<cv::Vec4i> hierarchy;
                findContours(region, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
                // TODO check epsilon parameter of approxPolyDP (set empirically) : we want more precission
                // if the region is very small because otherwise we'll loose all the convexities
                approxPolyDP(cv::Mat(contours[0]), contour_poly, cv::min(rect.width, rect.height) / 17.f, true);

                bool was_convex = false;
                int  num_inflexion_points = 0;
                for (size_t p = 0; p < contour_poly.size(); p++)
                {
                    int p_prev = p - 1;
                    int p_next = p + 1;
                    if (p_prev == -1)
                    {
                        p_prev = contour_poly.size() - 1;
                    }
                    if (p_next == contour_poly.size())
                    {
                        p_next = 0;
                    }

                    double angle_next = std::atan2((contour_poly[p_next].y - contour_poly[p].y), (contour_poly[p_next].x - contour_poly[p].x));
                    double angle_prev = std::atan2((contour_poly[p_prev].y - contour_poly[p].y), (contour_poly[p_prev].x - contour_poly[p].x));
                    if (angle_next < 0)
                    {
                        angle_next = 2.*PI + angle_next;
                    }

                    double angle = (angle_next - angle_prev);
                    if (angle > 2.*PI)
                    {
                        angle = angle - 2.*PI;
                    }
                    else if (angle < 0)
                    {
                        angle = 2.*PI + abs(angle);
                    }

                    if (p > 0)
                    {
                        if (((angle > PI) && (!was_convex)) || ((angle < PI) && (was_convex)))
                        {
                            num_inflexion_points++;
                        }
                    }
                    was_convex = (angle > PI);
                }

                cv::floodFill(region, cv::Point(0, 0), cv::Scalar(255), 0);
                int holes_area = region.cols*region.rows - countNonZero(region);

                int hull_area = 0;

                {

                    std::vector<cv::Point> hull;
                    cv::convexHull(contours[0], hull, false);
                    hull_area = static_cast<int>(contourArea(hull));
                }

                printf("%f,%f,%f\n",
                    static_cast<float>(holes_area) / pixelsFilled,
                    static_cast<float>(hull_area) / contourArea(contours[0]),
                    static_cast<float>(num_inflexion_points));
            }
            else
            {
                cv::floodFill(bwImage, seedPoint, zeroScalar);
            }

            delete[] crossings;
        }
    }

    return 0;
}


int main(int argc, char** argv)
{
    cv::Mat originalImage;

    if (argc == 1)
    {
        exit(0);
    }
    else
    {
        originalImage = cv::imread(argv[1], 0);
        originalImage = 255 - originalImage;
    }

    GroundTruth(originalImage);

    return 0;
}
