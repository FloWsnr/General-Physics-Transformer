traj_idx = 0
vel_mag = np.linalg.norm(vel[traj_idx, :, :, :, :], axis=-1)

# reduce dim with size 1 to 0
vel_mag = np.squeeze(vel_mag)

# transpose x and y
vel_mag = np.transpose(vel_mag, (0,  2, 1))

# Create normalized colormap for consistent color scaling
norm = Normalize(vmin=vel_mag.min(), vmax=vel_mag.max())

# Create frames
frames = []
for t in range(vel_mag.shape[0]):
    # Create figure and plot
    fig, ax = plt.subplots()
    im = ax.imshow(vel_mag[t,:,:])#, norm=norm)
    plt.colorbar(im)
    ax.set_title(f'Time step {t}')
    
    # Convert plot to image array
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
    frames.append(frame)
    plt.close()

# Save as GIF
output_path = data_path.parent
output_path = output_path / (str(data_path.stem) + "_vel.gif")
print(f"Writing {len(frames)} frames to {output_path}")
iio.imwrite(output_path, frames, fps=30)
print(f"Animation saved to {output_path}")