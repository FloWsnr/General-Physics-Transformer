export default function(eleventyConfig) {
  eleventyConfig.addPassthroughCopy("assets");

  return {
    dir: {
      input: ".",
      includes: "_includes",
      data: "_data",
      output: "_site"
    },
    eleventyExclude: ["claude.md"],
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk"
  };
}