import cv2
import numpy as np

#extracting frames:
for i in range(1,3):

    vid = cv2.VideoCapture(str(i)+".mp4")


    count, success = 0, True
    while success:
        success, image = vid.read() # Read frame
        if success: 
            cv2.imwrite("frame"+str(i)+"-"+str(count)+".jpg",image) # Save frame
            count += 1

    vid.release()

    # Background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=60,
        detectShadows=False
    )

    # Morphology kernel
    kernel = np.ones((3, 3), np.uint8)


    for j in range(0, count):
        filename = "frame"+str(i)+"-"+str(j)+".jpg"   # example: frame_001.png
        frame = cv2.imread(filename)

        if frame is None:
            print(f"Stopped at {filename} (file not found)")
            break


    # Stage 1: Background subtraction
    
        fg_mask = bg_subtractor.apply(frame)


    # Stage 2: Morphological cleaning
   
        clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # threshold to make mask cleaner
        _, clean = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)


    # Stage 3: Edge detection using derivatives
    
        gx = cv2.Sobel(clean, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(clean, cv2.CV_64F, 0, 1, ksize=3)

        mag = np.sqrt(gx**2 + gy**2)
        edges = np.uint8((mag > 45) * 255)

        def custom_hough_lines(edge_img):
            rows, cols = edge_img.shape

            thetas = np.deg2rad(np.arange(-90, 90, 1))
            diag_len = int(np.ceil(np.sqrt(rows * rows + cols * cols)))
            rhos = np.arange(-diag_len, diag_len + 1, 1)

            accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

            y_idxs, x_idxs = np.nonzero(edge_img)

            for i in range(len(x_idxs)):
                x = x_idxs[i]
                y = y_idxs[i]

                for t_idx in range(len(thetas)):
                    theta = thetas[t_idx]
                    rho = int(round(x * np.cos(theta) + y * np.sin(theta))) + diag_len
                    accumulator[rho, t_idx] += 1

            return accumulator, rhos, thetas


        def get_top_hough_lines(accumulator, rhos, thetas, num_lines=2):
            acc_copy = accumulator.copy()
            lines = []

            for _ in range(num_lines):
                idx = np.unravel_index(np.argmax(acc_copy), acc_copy.shape) #Finds the line (rho,theta)on which most of the points lie.
                rho_idx, theta_idx = idx

                rho = rhos[rho_idx]
                theta = thetas[theta_idx]
                votes = acc_copy[rho_idx, theta_idx]

                lines.append((rho, theta, votes))

        # suppress neighborhood around peak
                r0 = max(0, rho_idx - 4)
                r1 = min(acc_copy.shape[0], rho_idx + 5)
                t0 = max(0, theta_idx - 4)
                t1 = min(acc_copy.shape[1], theta_idx + 5)
                acc_copy[r0:r1, t0:t1] = 0

            return lines


        def draw_hough_line(img, rho, theta, color, thickness=2, scale=1.0):
            rho = rho / scale

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * (a)))
            pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * (a)))

            cv2.line(img, pt1, pt2, color, thickness)

        # Downscale for faster custom Hough
        scale = 0.25
        small_edges = cv2.resize(edges, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        accumulator, rhos, thetas = custom_hough_lines(small_edges)
        lines = get_top_hough_lines(accumulator, rhos, thetas, num_lines=2)

        output = frame.copy()

        for rho, theta, votes in lines:
            draw_hough_line(output, rho, theta, (0, 255, 0), 2, scale=scale)

        if len(lines) == 2:
            rho1, theta1, _ = lines[0]
            rho2, theta2, _ = lines[1]

        # medial axis = average of the two detected lines
            rho_mid = (rho1 + rho2) / 2
            theta_mid = (theta1 + theta2) / 2

        # draw medial axis in red
            draw_hough_line(output, rho_mid, theta_mid, (0, 0, 255), 3, scale=scale)

    # Show results
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Foreground Mask", fg_mask)
        cv2.imshow("Cleaned Mask", clean)
        cv2.imshow("Edges", edges)
        cv2.imshow("Small Edges", small_edges)
        cv2.imshow("Detected Lines", output)

    

        key = cv2.waitKey(30)
        if key == 27:   # press Esc to stop
            break

    

    cv2.destroyAllWindows()