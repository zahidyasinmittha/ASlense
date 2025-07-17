import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, RotateCcw, SkipBack, SkipForward, Maximize } from 'lucide-react';
import './VideoPlayer.css';

interface VideoPlayerProps {
  src: string;
  thumbnail?: string;
  title?: string;
  className?: string;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ src, thumbnail, title, className = '' }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [playbackRate, setPlaybackRate] = useState(1);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateTime = () => setCurrentTime(video.currentTime);
    const updateDuration = () => setDuration(video.duration);
    const handleEnded = () => setIsPlaying(false);

    video.addEventListener('timeupdate', updateTime);
    video.addEventListener('loadedmetadata', updateDuration);
    video.addEventListener('ended', handleEnded);

    return () => {
      video.removeEventListener('timeupdate', updateTime);
      video.removeEventListener('loadedmetadata', updateDuration);
      video.removeEventListener('ended', handleEnded);
    };
  }, []);

  // Auto-hide controls after 3 seconds
  useEffect(() => {
    let timer: number;
    if (isPlaying && showControls) {
      timer = window.setTimeout(() => setShowControls(false), 3000);
    }
    return () => window.clearTimeout(timer);
  }, [isPlaying, showControls]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
    setIsPlaying(!isPlaying);
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;

    video.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;

    const newVolume = parseFloat(e.target.value);
    video.volume = newVolume;
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const video = videoRef.current;
    const progressBar = progressRef.current;
    if (!video || !progressBar) return;

    const rect = progressBar.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const newTime = (clickX / rect.width) * duration;
    
    video.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const skipTime = (seconds: number) => {
    const video = videoRef.current;
    if (!video) return;

    video.currentTime = Math.max(0, Math.min(duration, currentTime + seconds));
  };

  const resetVideo = () => {
    const video = videoRef.current;
    if (!video) return;

    video.currentTime = 0;
    setCurrentTime(0);
    setIsPlaying(false);
    video.pause();
  };

  const changePlaybackRate = (rate: number) => {
    const video = videoRef.current;
    if (!video) return;

    video.playbackRate = rate;
    setPlaybackRate(rate);
  };

  const toggleFullscreen = () => {
    const video = videoRef.current;
    if (!video) return;

    if (!isFullscreen) {
      video.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
    setIsFullscreen(!isFullscreen);
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const progressPercentage = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div 
      className={`video-player ${className}`}
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => !isPlaying || setShowControls(false)}
    >
      <video
        ref={videoRef}
        className="video-player__video"
        poster={thumbnail}
        onClick={togglePlay}
        onError={(e) => console.error("Video playback error:", e)}
      >
        <source src={src} type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      {showControls && (
        <div className="video-player__controls">
          {/* Progress Bar */}
          <div 
            ref={progressRef}
            className="video-player__progress"
            onClick={handleSeek}
          >
            <div 
              className="video-player__progress-fill"
              style={{ width: `${progressPercentage}%` }}
            />
          </div>

          {/* Main Controls */}
          <div className="video-player__main-controls">
            <div className="video-player__left-controls">
              <button onClick={togglePlay} className="video-player__control-btn">
                {isPlaying ? <Pause size={20} /> : <Play size={20} />}
              </button>
              
              <button onClick={() => skipTime(-10)} className="video-player__control-btn">
                <SkipBack size={18} />
              </button>
              
              <button onClick={() => skipTime(10)} className="video-player__control-btn">
                <SkipForward size={18} />
              </button>
              
              <button onClick={resetVideo} className="video-player__control-btn">
                <RotateCcw size={18} />
              </button>

              <div className="video-player__volume">
                <button onClick={toggleMute} className="video-player__control-btn">
                  {isMuted || volume === 0 ? <VolumeX size={18} /> : <Volume2 size={18} />}
                </button>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={isMuted ? 0 : volume}
                  onChange={handleVolumeChange}
                  className="video-player__volume-slider"
                />
              </div>

              <div className="video-player__time">
                {formatTime(currentTime)} / {formatTime(duration)}
              </div>
            </div>

            <div className="video-player__right-controls">
              <div className="video-player__speed">
                <select 
                  value={playbackRate} 
                  onChange={(e) => changePlaybackRate(parseFloat(e.target.value))}
                  className="video-player__speed-select"
                >
                  <option value={0.5}>0.5x</option>
                  <option value={0.75}>0.75x</option>
                  <option value={1}>1x</option>
                  <option value={1.25}>1.25x</option>
                  <option value={1.5}>1.5x</option>
                  <option value={2}>2x</option>
                </select>
              </div>

              <button onClick={toggleFullscreen} className="video-player__control-btn">
                <Maximize size={18} />
              </button>
            </div>
          </div>
        </div>
      )}

      {title && (
        <div className="video-player__title">
          {title}
        </div>
      )}
    </div>
  );
};

export default VideoPlayer;
