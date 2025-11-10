import React from "react";
import { motion } from "framer-motion";

const Hero = ({ title = "Start Saving with Bargain Bee!", subtitle = "Search by brand, product, grocery, location " }) => {
  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { duration: 1.5, when: "beforeChildren", staggerChildren: 0.3 },
    },
  };

  const textVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0 },
  };

  const backgroundVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 1 } },
  };

  return (
    <motion.div
      className="relative h-[40vh] sm:h-[50vh] lg:h-[60vh] rounded-none sm:rounded-2xl shadow-lg mb-8 overflow-hidden font-['Nunito']"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Gradient Background */}
      <motion.div
        className="absolute inset-0 animate-pulse"
        style={{
          background: "linear-gradient(135deg, #d5e6fa, #FFD27F, #b0c8f4)",
        }}
        variants={backgroundVariants}
      ></motion.div>

      {/* Animated Wave */}
      <div className="absolute bottom-0 w-full overflow-hidden">
        <svg
          className="relative w-[200%] h-full filter blur-lg animate-wave"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 1440 320"
        >
          <defs>
            <linearGradient id="waveGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style={{ stopColor: "#d5e6fa", stopOpacity: 1 }} />
              <stop offset="50%" style={{ stopColor: "#FFD27F", stopOpacity: 1 }} />
              <stop offset="100%" style={{ stopColor: "#b0c8f4", stopOpacity: 1 }} />
            </linearGradient>
          </defs>
          <path
            fill="url(#waveGradient)"
            d="M0,300C60,290,120,250,180,250C240,250,300,280,360,270C420,260,480,220,540,230C600,240,660,300,720,290C780,280,840,230,900,220C960,210,1020,260,1080,270C1140,280,1200,230,1260,240C1320,250,1380,290,1440,300L1440,320L0,320Z"
          ></path>
        </svg>
      </div>

      {/* Text Content */}
      <motion.div
        className="relative z-10 flex flex-col items-center justify-center h-full text-center text-gray-800 px-4"
        variants={textVariants}
      >
        {/* Title */}
        <motion.h1
          className="font-bold mt-8"
          style={{ fontSize: "clamp(1.5rem, 3vw, 2.5rem)" }}
          variants={textVariants}
        >
          {title}
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          className="mt-3 text-gray-500"
          style={{ fontSize: "clamp(1rem, 2vw, 1.5rem)" }}
          variants={textVariants}
        >
          {subtitle.split(",").map((word, idx) => (
            <span key={idx} className={idx > 0 ? "text-gray-800" : ""}>
              {idx > 0 && ","} {word.trim()}
            </span>
          ))}
        </motion.p>
      </motion.div>
    </motion.div>
  );
};

export default Hero;
