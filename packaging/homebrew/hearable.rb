# Homebrew formula for a third-party tap (e.g. `brew tap alonecandies/hearable`).
#
# DRAFT — two things must be resolved before this installs cleanly:
#   1. A tagged release (fill `url` + `sha256`), or rely on the `head` build below.
#   2. sherpa-onnx's build.rs downloads prebuilt native libs from GitHub at build time, but
#      Homebrew's build sandbox blocks network. Declare those libs as a `resource` and point
#      SHERPA_ONNX_ARCHIVE_DIR / SHERPA_ONNX_LIB_DIR at them (sketched in `install`), or
#      vendor them. Until then, build from source outside Homebrew (see scripts/bundle-macos.sh).
class Hearable < Formula
  desc "Real-time on-device captioning + speaker-ID overlay for Deaf/HoH users"
  homepage "https://github.com/alonecandies/hearable"
  license "MIT"
  head "https://github.com/alonecandies/hearable.git", branch: "master"

  # stable do
  #   url "https://github.com/alonecandies/hearable/archive/refs/tags/v0.1.0.tar.gz"
  #   sha256 "FILL_ON_RELEASE"
  # end

  depends_on "rust" => :build

  def install
    # If/when sherpa libs are a declared resource, stage and point the env at them:
    #   resource("sherpa-onnx-libs").stage { ... }
    #   ENV["SHERPA_ONNX_LIB_DIR"] = "#{buildpath}/sherpa-libs"
    system "cargo", "build", "--release", "--features", "live"
    bin.install "target/release/hearable"
  end

  def caveats
    <<~EOS
      hearable needs microphone access and ML model files at runtime.
      Fetch models:   #{opt_prefix}/../scripts/fetch-models.sh ./models   (or download manually)
      Run:            hearable run --models ./models
      On macOS, grant Microphone permission when prompted. For the cleanest permission
      experience, use the .app bundle (scripts/bundle-macos.sh) rather than the bare binary.
    EOS
  end

  test do
    assert_match "hearable", shell_output("#{bin}/hearable --help")
  end
end
