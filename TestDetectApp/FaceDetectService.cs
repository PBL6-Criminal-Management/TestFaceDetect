using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using Emgu.CV.CvEnum;
using System.Drawing;

namespace TestDetectApp
{
    public class FaceDetectService
    {
        EigenFaceRecognizer recognizer = new EigenFaceRecognizer();
        readonly CascadeClassifier faceCasacdeClassifier = new CascadeClassifier(@".\haarcascade_frontalface_default.xml");
        
        List<Image<Gray, Byte>> TrainedFaces = new List<Image<Gray, byte>>();
        List<int> PersonsLabes = new List<int>();
        Dictionary<int, string> PersonsNames = new Dictionary<int, string> { { 1, "Hoan" }, { 2, "Luan" } };

        bool isTrained = false;
        readonly int ThresholdForDetectedFacesImage = 10700;
        readonly int ThresholdForNonDetectedFacesImage = 2000;
        readonly Size allowedFaceMinimizeSize = new Size(70, 70);

        static readonly string detectBasePath = $"{Directory.GetCurrentDirectory()}/Detect";

        readonly string modelPath = $"{detectBasePath}/Model/model.xml";


        public FaceDetectService()
        {
            if (!Directory.Exists(detectBasePath))
            {
                Directory.CreateDirectory(detectBasePath);
            }
            if (File.Exists(modelPath))
            {
                //Load model from file
                recognizer.Read(modelPath);
                isTrained = true;
            }
        }

        public string FaceDetect(IFormFile file, bool enableSaveImage)
        {
            if (isTrained)
            {
                string res = "";
                try
                {
                    // Read the content of the IFormFile into a byte array
                    using (var ms = new MemoryStream())
                    {
                        file.CopyTo(ms);
                        var fileBytes = ms.ToArray();

                        //Step: 1 Create a Mat object from the byte array
                        Mat imageMat = new Mat();
                        CvInvoke.Imdecode(fileBytes, ImreadModes.Color, imageMat);

                        // Convert Mat to Image<Bgr, Byte>
                        Image<Bgr, Byte> imageFrame = imageMat.ToImage<Bgr, Byte>();

                        //Step 2: Face Detection
                        //Convert from Bgr to Gray Image
                        Mat grayImage = new Mat();
                        CvInvoke.CvtColor(imageFrame, grayImage, ColorConversion.Bgr2Gray);

                        //Enhance the image to get better result
                        CvInvoke.EqualizeHist(grayImage, grayImage);

                        //Find faces
                        Rectangle[] faces = faceCasacdeClassifier.DetectMultiScale(grayImage, 1.1, 3, allowedFaceMinimizeSize, Size.Empty);

                        //If faces detected
                        if (faces.Length > 0)
                        {
                            foreach (var face in faces)
                            {
                                //Draw square arou2nd each face 
                                CvInvoke.Rectangle(imageFrame, face, new Bgr(Color.Red).MCvScalar, 2);

                                //Step 3: Add Person 
                                Image<Bgr, Byte> resultImage = imageFrame.Convert<Bgr, Byte>();
                                resultImage.ROI = face;

                                //Step 4: Export input image (option)
                                if (enableSaveImage)
                                    SaveImage(imageFrame.Copy(), @"\InputImage", "Input");

                                enableSaveImage = false;

                                // Step 5: Recognize the face 
                                Image<Gray, Byte> grayFaceResult = resultImage.Convert<Gray, Byte>().Resize(200, 200, Inter.Cubic);
                                CvInvoke.EqualizeHist(grayFaceResult, grayFaceResult);
                                var result = recognizer.Predict(grayFaceResult);

                                Console.WriteLine($"Label: {result.Label}, Distance: {result.Distance}");

                                //Here results found known faces
                                if (result.Label != -1 && result.Distance < ThresholdForDetectedFacesImage)
                                {
                                    res = "id: " + result.Label + " name: " + PersonsNames[result.Label];
                                    CvInvoke.PutText(imageFrame, PersonsNames[result.Label], new Point(face.X - 2, face.Y - 2),
                                        FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                                    CvInvoke.Rectangle(imageFrame, face, new Bgr(Color.Green).MCvScalar, 2);
                                }
                                //here results did not found any know faces
                                else
                                {
                                    res = "Unknown";
                                    CvInvoke.PutText(imageFrame, "Unknown", new Point(face.X - 2, face.Y - 2),
                                        FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                                    CvInvoke.Rectangle(imageFrame, face, new Bgr(Color.Red).MCvScalar, 2);
                                }
                            }
                        }
                        else
                        {
                            Image<Bgr, Byte> resultImage = imageFrame.Convert<Bgr, Byte>();

                            //Export input image
                            if (enableSaveImage)
                                SaveImage(imageFrame.Copy(), @"\InputImage", "Input");

                            enableSaveImage = false;

                            // Step 5: Recognize the face 
                            Image<Gray, Byte> grayFaceResult = resultImage.Convert<Gray, Byte>().Resize(200, 200, Inter.Cubic);
                            CvInvoke.EqualizeHist(grayFaceResult, grayFaceResult);
                            var result = recognizer.Predict(grayFaceResult);

                            Console.WriteLine($"Label: {result.Label}, Distance: {result.Distance}");


                            if (result.Label != -1 && result.Distance < ThresholdForNonDetectedFacesImage)
                            {
                                res = "id: " + result.Label + " name: " + PersonsNames[result.Label];
                                CvInvoke.PutText(imageFrame, PersonsNames[result.Label], new Point(30, 30),
                                    FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                            }
                            else
                            {
                                res = "Unknown";
                                CvInvoke.PutText(imageFrame, "Unknown", new Point(30, 30),
                                    FontFace.HersheyComplex, 1.0, new Bgr(Color.Orange).MCvScalar);
                            }
                        }

                        //export to png
                        SaveImage(imageFrame, @"\DetectResult", "Result");
                    }

                }
                catch (Exception ex)
                {
                    return $"Internal server error: {ex.Message}";
                }

                return res;
            }
            else
                return "AI model hasn't trained yet!";
        }


        //Re-train
        //Step 4: train Images .. we will use the saved images
        public string TrainImagesFromDir()
        {
            TrainedFaces.Clear();
            PersonsLabes.Clear();
            try
            {
                string path = $@"{detectBasePath}\TrainedImages";
                if (!Directory.Exists(path))
                    return "There aren't any image to train!";

                string[] folders = Directory.GetDirectories(path);

                foreach (var folder in folders)
                {
                    string[] images = Directory.GetFiles(folder);
                    foreach(string image in images)
                    {
                        Image<Bgr, byte> resultImage = new Image<Bgr, byte>(image);

                        Mat grayImage = new Mat();
                        CvInvoke.CvtColor(resultImage, grayImage, ColorConversion.Bgr2Gray);
                        CvInvoke.EqualizeHist(grayImage, grayImage);

                        Rectangle[] faces = faceCasacdeClassifier.DetectMultiScale(grayImage, 1.1, 3, allowedFaceMinimizeSize, Size.Empty);

                        //If faces detected
                        if (faces.Length > 0)
                        {
                            var face = faces[0];

                            //Draw square arou2nd each face 
                            CvInvoke.Rectangle(resultImage, face, new Bgr(Color.Red).MCvScalar, 2);

                            resultImage = resultImage.Convert<Bgr, Byte>();
                            resultImage.ROI = face;

                            //Export input image
                            //SaveImage(resultImage, @"\ImageToTrain", $"ImageToTrain");
                        }

                        Image<Gray, byte> trainedImage = resultImage.Convert<Gray, Byte>().Resize(200, 200, Inter.Cubic);

                        CvInvoke.EqualizeHist(trainedImage, trainedImage);

                        //SaveImage(trainedImage, @"\TrainGrayImage", "Train");
                        TrainedFaces.Add(trainedImage);
                        PersonsLabes.Add(Convert.ToInt32(folder.Split('\\').Last()));
                    }
                }

                if (TrainedFaces.Count() > 0)
                {   
                    recognizer = new EigenFaceRecognizer(100, double.PositiveInfinity);
                    recognizer.Train(TrainedFaces.Select(img => img.Mat).ToArray(), PersonsLabes.ToArray());

                    // Save model to file
                    string? directoryPath = Path.GetDirectoryName(modelPath);
                    if (directoryPath != null && !Directory.Exists(directoryPath))
                        Directory.CreateDirectory(directoryPath);

                    recognizer.Write(modelPath);

                    isTrained = true;
                    Console.WriteLine(TrainedFaces.Count);

                    return "AI model is trained successfully!";
                }
                else
                {
                    isTrained = false;
                    return "There aren't any images to train!";
                }
            }
            catch (Exception ex)
            {
                isTrained = false;
                Console.WriteLine("Error in Train Images: " + ex.Message);
                return "Get error when train AI model!";
            }

        }

        static void SaveImage<TColor>(Image<TColor, byte> resultImage, string path, string imageType) where TColor : struct, IColor
        {
            string savePath = detectBasePath + path;
            if (!Directory.Exists(savePath))
                Directory.CreateDirectory(savePath);

            //to avoid hang GUI we will create a new task
            Task.Factory.StartNew(() =>
            {
                if (resultImage != null)
                    resultImage.Save(savePath + @"\" + imageType + "_" + Guid.NewGuid().ToString() + ".png");
            });
        }
    }
}
