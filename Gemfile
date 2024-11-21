source "https://rubygems.org"

gem "jekyll", "~> 4.3.3"
gem "jekyll-theme-chirpy", "~> 6.4.2"

# Plugins
group :jekyll_plugins do
  gem "jekyll-paginate"
  gem "jekyll-archives"
  gem "jekyll-sitemap"
  gem "jekyll-feed"
end

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
gem "tzinfo-data", platforms: [:mingw, :mswin, :x64_mingw, :jruby]

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :mswin, :x64_mingw]

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds since newer versions of the gem
# do not have a Java counterpart.
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby] 