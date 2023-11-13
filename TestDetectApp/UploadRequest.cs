namespace TestDetectApp
{
    public class UploadRequest
    {
        public IFormFile File { get; set; } = default!;
        public bool IsSave { get; set; } = default!;
    }
}
