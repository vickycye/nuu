# nuu

## Inspiration
We've all been there: standing in IKEA, staring at a beautiful couch, wondering "will this actually fit in my apartment?" You pull out your phone, squinting at blurry photos of your living room, trying to mentally calculate if that couch would fit, and if the color is right. One of us is an avid floorplan enthusiast who sketches them in her free time, and the other is currently taking a computer vision class and kept thinking: why doesn't this technology exist yet?

As college students constantly moving between dorms and apartments, we're perpetually furniture shopping on tight budgets. We can't afford expensive mistakes like buying a desk that's too big, a rug in the wrong color that we will despise for the rest of the year, or a bookshelf that blocks the only outlet. We realized that what we really needed was a way to carry our space with us, like a digital shell we could reference anytime and anywhere.

The inspiration of our product's name and logo is from Nuu, an adorable little snail from Hollow Knight: Silksong. Nuu carries her home on her back whenever she goes, a perfect metaphor for what we wanted to create. You room becomes portable, accessible from your pocket, ready to help you make confident decisions about your space. 

## What it does
Nuu transforms any room into an interactive 3D model that you can pull up on your phone or laptop. Here's how it works:
1. Scan it: Walk around your room while recording a short video on your phone (10-30 seconds).
2. Process: Nuu analyzes the video, estimating depth, tracking camera movement, and identifying room structure.
3. Explore: Get a full 3D reconstruction of your space that you can navigate with first-person controls (WASD + mouse, like a video game on your laptop, or touch controls on your phone, still being implemented!)
4. Plan it & buy it: Take your virtual room with you to furniture stores, compare dimensions in real-time, and visualize how new pieces will look. 

## How we built it
**Frontend**: Three.js for 3D rendering and navigation, React + Vite for UI, CSS, Vercel (planned) for deployment.
**Backend**: Python + FastAPI for the processing pipeline, OpenCV for video frame extraction, MiDaS for monocular depth estimation, COLMAP for camera pose estimation and structure-from-motion, Open3D for point cloud processing and mesh generation, and planning on deploying backend with render/railway.

We started with a flowchart diagram on a whiteboard then divided responsibilities: one teammate focused on creating an intuitive Three.js viewer with smooth navigation controls, while the other built the computer vision pipeline from video input to 3D mesh output. We leveraged AI assistance throughout to accelerate learning new frameworks and to debug complex issues. 

## Challenges we ran into
**For the frontend**, the steepest learning curve was Three.js, as neither of us had touched 3D graphics before this hackathon. Loading and rending .glb files seemed simple in tutorials but we hit a lot of issues related to camera positioning and coordinate system mismatches. After a few hours of debugging and reading documentation, we finally got our first mesh to appear on screen.

**For the backend**, the 3D reconstruction part was the hardest. While we successfully got depth estimation working on the first or second try, and camera post estimation running, translating these to a clean mesh was brutal. Our early attempts produced...well, let's call them "abstract art". They looked more like crystal fragments than rooms. 

The core issue is that COLMAP is actually incredibly sensitive to camera parameters, lighting conditions, and video quality. Phone videos with motion blur or low texture would fail to reconstruct. When it did work, the point cloud was often noisy, requiring extensive filtering. Converting noisy point clouds to clean meshes using Poisson reconstruction also felt like an impossible task while we were bouncing between tuning the parameters and learning how the math works behind it.

## What we learned
On the technical side, we disovered that Three.js is powerful but also unforgiving. 3D graphics programming requires thinking in completely different terms about coordinate systems, matrices, shaders, and camera projections, where small mistakes will cascade into insane visual bugs. We also learned that CV is genuinely difficult; even with a bunch of pre-trained models, making them work together reliably is challenging, and real-world data like shaky phone videos and varied lighting breaks algorithms in unexpected ways. 

Beyond technical skills, we learned about scope management. Our original wishlist of furniture detection, AR overlays, and export features would sink the project, so cutting scope to focus on core functionality was the best decision we made. Cross-domain collaboration taught us that frontend and backend engineers must constantly communicate about data formats and API design. We also discovered that AI-assisted development can dramatically accelerate learning new frameworks and debugging issues, though deep understanding is still essential for integration. Most importantly, we proved to ourselves that we can learn complex technologies under time pressure, and that sometimes, "good enough" truly is good enough. 

## What's next for Nuu
Immediately after the hackathon ends, we're focused on fixing the reconstruction algorithm by tuning COLMAP parameters more, adding better preprocessing steps, and implementing improvements to eliminate those abstract blob meshes. We want to add furniture recognition using object detection models like YOLOv8 to automatically identify and label furniture in the scene, along with measurement tools that overlay dimensions so users can measure distances in their virtual room. 

Our longer-term vision transforms Nuu into a comprehensive spatial intelligence platform. We imagine a furniture placement mode where you can scan furniture from stores and drag-and-drop virtual pieces into your room model to visualize before buying, complemented by AR view that overlays the 3D reconstruction onto your phone's camera for augmented reality exploration. We want to enable collaborative shopping where you share your room model with friends and family for second opinions, and even implement AI-powered style matching that suggests furniture matching your room's aesthetic. Ultimately, we dream of helping people not just shop for furniture, but plan renovations, estimate moving costs, create virtual staging for real estate, and make their physical spaces truly portable and accessible from anywhere. Like our namesake snail carrying her home, we want everyone to carry their space with them.
