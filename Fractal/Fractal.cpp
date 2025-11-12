#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#include <sycl/sycl.hpp>
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

#include <vector>
#include <complex>

using namespace std::literals::complex_literals;
typedef std::complex<float> complex;

constexpr size_t WIDTH = 1024, HEIGHT = 768;
constexpr size_t ITERATION_COUNT = 20;
constexpr uint8_t FILL_INTENSITY = 255;

static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const& e : e_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (std::exception const& e) {
#if _DEBUG
            std::cout << "Failure" << std::endl;
#endif
            std::terminate();
        }
    }
    };

static void newton(
    sycl::queue& q,
    const complex a, const complex b, const complex c,
    std::vector<complex>& buf, std::vector<uint8_t>& out,
    const size_t w, const size_t h, const double unit,
    const double left, const double top,
    const size_t iterations
) {
    memset(out.data(), 0, out.size());
    const complex sum = a + b + c;
    const complex prod_sum = a * b + a * c + b * c;
    const complex prod = a * b * c;
    sycl::range<2> num_items{ h, w };
    sycl::buffer<complex, 2> buf_vec{ buf.data(), num_items };
    sycl::buffer<uint8_t, 1> buf_out(out.data(), sycl::range<1>(h * w * 4));
    q.submit([&](sycl::handler& h) {
        sycl::accessor vec(buf_vec, h, sycl::write_only);
        h.parallel_for(num_items, [=](auto i) {
            const double y = top - unit * i[0];
            const double x = left + unit * i[1];
            const complex v{ x, y };
            vec[i] = v;
            });
        });
    for (size_t i = 0; i < iterations; ++i) {
        q.submit([&](sycl::handler& h) {
            sycl::accessor vec(buf_vec, h, sycl::read_write);
            h.parallel_for(num_items, [=](auto i) {
                const complex x = vec[i];
                const complex sqr = x * x;
                const complex v = sqr * x - sum * sqr + prod_sum * x - prod;
                const complex d = sqr * 3.0f - sum * x * 2.0f + prod_sum;
                vec[i] = x - v / d;
                });
            });
    }
    q.submit([&](sycl::handler& h) {
        sycl::accessor vec(buf_vec, h, sycl::read_only);
        sycl::accessor out(buf_out, h, sycl::write_only);
        h.parallel_for(num_items, [=](auto i) {
            const complex x = vec[i];
            const complex da = x - a;
            const complex db = x - b;
            const complex dc = x - c;
            const double da_hyp = da.real() * da.real() + da.imag() * da.imag();
            const double db_hyp = db.real() * db.real() + db.imag() * db.imag();
            const double dc_hyp = dc.real() * dc.real() + dc.imag() * dc.imag();
            const size_t index = (i[0] * w + i[1]) * 4;
            if (da_hyp <= db_hyp && da_hyp <= dc_hyp) out[index] = FILL_INTENSITY;
            else if (db_hyp <= da_hyp && db_hyp <= dc_hyp) out[index + 1] = FILL_INTENSITY;
            else out[index + 2] = FILL_INTENSITY;
            });
        });
    q.wait();
}

#if FPGA_EMULATOR
// Intel extension: FPGA emulator selector on systems without FPGA card.
const auto device_selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif FPGA_SIMULATOR
// Intel extension: FPGA simulator selector on systems without FPGA card.
const auto device_selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
// Intel extension: FPGA selector on systems with FPGA card.
const auto device_selector = sycl::ext::intel::fpga_selector_v;
#else
// The default device selector will select the most performant device.
const auto device_selector = sycl::default_selector_v;
#endif

static double coords_to_pix_x(const double x, const double left, const double unit_width) {
    return (x - left) / unit_width * static_cast<double>(WIDTH);
}

static double coords_to_pix_y(const double y, const double top, const double unit_height) {
    return (top - y) / unit_height * static_cast<double>(HEIGHT);
}

static double pix_to_coords_x(const double x, const double left, const double unit_width) {
    return x / static_cast<double>(WIDTH) * unit_width + left;
}

static double pix_to_coords_y(const double y, const double top, const double unit_height) {
    return top - y / static_cast<double>(HEIGHT) * unit_height;
}

static bool is_in_range(const double x1, const double y1, const double x2, const double y2) {
    return std::abs(x1 - x2) < 5.0 && std::abs(y1 - y2) < 5.0;
}

int main(int argc, char* argv[]) {
    SDL_SetMainReady();
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        return 1;
    }

    sycl::queue queue(device_selector, exception_handler);

    constexpr size_t buffer_size = WIDTH * HEIGHT * 4;

    std::vector<uint8_t> pixels(buffer_size);
    std::vector<complex> buffer(buffer_size);

    bool running = true;
    bool changed = true;
    size_t frame_count = 0;
    uint32_t last_time = SDL_GetTicks();
    size_t iteration_count = ITERATION_COUNT;
    complex a = -2.0f + 1.0if, b = 2.0f + 2.0if, c = -1.0f - 2.0if;
    double unit_width = 10.0, left = -5.0, top = 4.0;
    double unit_height = unit_width * (static_cast<double>(HEIGHT) / static_cast<double>(WIDTH));
    SDL_FRect roots[3] = { {0.0, 0.0, 10.0, 10.0}, {0.0, 0.0, 10.0, 10.0}, {0.0, 0.0, 10.0, 10.0} };
    size_t point_dragging_index = 0;
    bool position_dragging = false;

    SDL_Event event;

    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;
    SDL_Texture* texture = NULL;

    window = SDL_CreateWindow(
        "Fractal",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        WIDTH,
        HEIGHT,
        SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_RESIZABLE
    );
    if (window == NULL) goto end;

    renderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    if (renderer == NULL) goto end;

    texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        WIDTH,
        HEIGHT
    );
    if (texture == NULL) goto end;

    while (running) {
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_WINDOWEVENT_RESIZED:
                changed = true;
                break;
            case SDL_MOUSEBUTTONUP:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    point_dragging_index = 0;
                    std::cout << "a: " << a << std::endl;
                    std::cout << "b: " << b << std::endl;
                    std::cout << "c: " << c << std::endl;
                }
                else if (event.button.button == SDL_BUTTON_RIGHT) position_dragging = false;
                break;
            case SDL_MOUSEBUTTONDOWN:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    if (is_in_range(roots[0].x + 5.0, roots[0].y + 5.0, event.button.x, event.button.y)) {
                        changed = true;
                        point_dragging_index = 1;
                    }
                    else if (is_in_range(roots[1].x + 5.0, roots[1].y + 5.0, event.button.x, event.button.y)) {
                        changed = true;
                        point_dragging_index = 2;
                    }
                    else if (is_in_range(roots[2].x + 5.0, roots[2].y + 5.0, event.button.x, event.button.y)) {
                        changed = true;
                        point_dragging_index = 3;
                    }
                }
                else if (event.button.button == SDL_BUTTON_RIGHT) {
                    position_dragging = true;
                }
                break;
            case SDL_MOUSEMOTION:
                if (point_dragging_index) {
                    const double x = pix_to_coords_x(event.motion.x, left, unit_width);
                    const double y = pix_to_coords_y(event.motion.y, top, unit_height);
                    changed = true;
                    switch (point_dragging_index) {
                    case 1:
                        a = complex(x, y);
                        break;
                    case 2:
                        b = complex(x, y);
                        break;
                    case 3:
                        c = complex(x, y);
                        break;
                    }
                }
                else if (position_dragging) {
                    left -= event.motion.xrel * unit_width / static_cast<double>(WIDTH);
                    top += event.motion.yrel * unit_height / static_cast<double>(HEIGHT);
                    changed = true;
                }
                break;
            case SDL_MOUSEWHEEL:
                if (event.wheel.y == 0) break;
                changed = true;
                top -= unit_height * event.wheel.y * 0.025;
                left += unit_width * event.wheel.y * 0.025;
                unit_width *= std::pow(0.95, event.wheel.y);
                unit_height = unit_width * (static_cast<double>(HEIGHT) / static_cast<double>(WIDTH));
                break;
            case SDL_KEYDOWN:
                if (event.key.keysym.sym == SDLK_r) {
                    a = -2.0 + 1.0i, b = 2.0 + 2.0i, c = -1.0 - 2.0i;
                    unit_width = 10.0, left = -5.0, top = 4.0;
                    unit_height = unit_width * (static_cast<double>(HEIGHT) / static_cast<double>(WIDTH));
                    point_dragging_index = 0;
                    position_dragging = false;
                    iteration_count = 20;
                    changed = true;
                    std::cout << "Reset" << std::endl;
                }
                else if (event.key.keysym.sym == SDLK_UP) {
                    ++iteration_count;
                    changed = true;
                    std::cout << "Iterations: " << iteration_count << std::endl;
                }
                else if (event.key.keysym.sym == SDLK_DOWN) {
                    if (iteration_count == 0) iteration_count = 1;
                    --iteration_count;
                    changed = true;
                    std::cout << "Iterations: " << iteration_count << std::endl;
                }
                break;
            }
        }
        const uint8_t* keyboard = SDL_GetKeyboardState(NULL);
        if (keyboard[SDL_SCANCODE_A] && keyboard[SDL_SCANCODE_D]) {}
        else if (keyboard[SDL_SCANCODE_A]) {
            changed = true;
            left -= unit_width * 0.01;
        }
        else if (keyboard[SDL_SCANCODE_D]) {
            changed = true;
            left += unit_width * 0.01;
        }
        if (keyboard[SDL_SCANCODE_W] && keyboard[SDL_SCANCODE_S]) {}
        else if (keyboard[SDL_SCANCODE_S]) {
            changed = true;
            top -= unit_width * 0.01;
        }
        else if (keyboard[SDL_SCANCODE_W]) {
            changed = true;
            top += unit_width * 0.01;
        }
        if ((keyboard[SDL_SCANCODE_LSHIFT] || keyboard[SDL_SCANCODE_RSHIFT]) && keyboard[SDL_SCANCODE_SPACE]) {}
        else if (keyboard[SDL_SCANCODE_LSHIFT] || keyboard[SDL_SCANCODE_RSHIFT]) {
            changed = true;
            top -= unit_height * 0.025;
            left += unit_width * 0.025;
            unit_width *= 0.95;
            unit_height = unit_width * (static_cast<double>(HEIGHT) / static_cast<double>(WIDTH));
        }
        else if (keyboard[SDL_SCANCODE_SPACE]) {
            changed = true;
            top += unit_height * 0.025;
            left -= unit_width * 0.025;
            unit_width *= 1.05;
            unit_height = unit_width * (static_cast<double>(HEIGHT) / static_cast<double>(WIDTH));
        }
        if (changed) {
            newton(queue, a, b, c, buffer, pixels, WIDTH, HEIGHT, unit_width / static_cast<double>(WIDTH), left, top, iteration_count);
            SDL_UpdateTexture(texture, nullptr, pixels.data(), WIDTH * 4);
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, nullptr, nullptr);
            SDL_SetRenderDrawColor(renderer, FILL_INTENSITY, FILL_INTENSITY, FILL_INTENSITY, 255);
            roots[0].x = coords_to_pix_x(a.real(), left, unit_width) - 5.0;
            roots[0].y = coords_to_pix_y(a.imag(), top, unit_height) - 5.0;
            roots[1].x = coords_to_pix_x(b.real(), left, unit_width) - 5.0;
            roots[1].y = coords_to_pix_y(b.imag(), top, unit_height) - 5.0;
            roots[2].x = coords_to_pix_x(c.real(), left, unit_width) - 5.0;
            roots[2].y = coords_to_pix_y(c.imag(), top, unit_height) - 5.0;
            SDL_RenderDrawRectsF(renderer, roots, 3);
            changed = false;
        }
        SDL_RenderPresent(renderer);
        const uint32_t this_time = SDL_GetTicks();
        ++frame_count;
        if (this_time - last_time >= 5000) {
            last_time = this_time;
            std::cout << "FPS: " << frame_count / 5 << std::endl;
            frame_count = 0;
        }
    }

end:
    if (texture != NULL) SDL_DestroyTexture(texture);
    if (renderer != NULL) SDL_DestroyRenderer(renderer);
    if (window != NULL) SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
