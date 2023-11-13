using Microsoft.AspNetCore.Mvc;

namespace TestDetectApp.Controllers
{
    [ApiController]
    [Route("api/facedetect")]
    public class FaceDetect : Controller
    {

        private readonly FaceDetectService _fds;

        public FaceDetect(FaceDetectService fds)
        {
            _fds = fds;
        }

        // GET: FaceDetect
        [HttpGet]
        [Route("Retrain")]
        public ActionResult<string> Retrain()
        {
            return _fds.TrainImagesFromDir();
        }     

        // GET: FaceDetect
        [HttpPost]
        [Route("FaceDetect")]
        [Consumes("multipart/form-data")]
        public ActionResult<string> FaceDetectAPI([FromForm] UploadRequest request)
        {
            return _fds.FaceDetect(request.File, request.IsSave);
        }     
    }
}
