use std::any::Any;
use std::io::Cursor;
use std::sync::Arc;

use rand::prelude::*;
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{debug_handler, Router};
use axum::routing::get;
use eyre::{bail, Result};
use fatline_rs::{HubService, HubServiceClient};
use fatline_rs::users::UserService;
use image::{DynamicImage, ImageOutputFormat};
use image::imageops::FilterType;
use rand::distributions::Standard;
use rand::rngs::adapter::ReseedingRng;
use rand::rngs::mock::StepRng;
use rand::rngs::OsRng;
use rand::thread_rng;
use rand_chacha::ChaCha8Rng;
use rust_faces::{BlazeFace, BlazeFaceParams, FaceDetector, FaceDetectorBuilder, Nms, ToArray3};
use rust_faces::FaceDetection::BlazeFace640;
use rust_faces::priorboxes::PriorBoxesParams;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

#[derive(Clone)]
struct AppState {
    face_detector: Arc<Mutex<Box<dyn FaceDetector>>>,
    client: Arc<Mutex<HubService>>
}

type FaceState = Arc<Mutex<Box<dyn FaceDetector>>>;

#[tokio::main]
async fn main() -> Result<()> {

    let state: FaceState = Arc::new(Mutex::new(get_detector()));
    let url = dotenv::var("HUB_URL")?;
    let bind = dotenv::var("BIND_URL")?;
    println!("{url}");
    let client = HubService::connect(url).await?;

    let router = Router::new()
        .route("/face/:fid", get(get_face_pic))
        .with_state(AppState{
            face_detector: state,
            client: Arc::new(Mutex::new(client))
        });

    let listener = TcpListener::bind(bind).await?;

    axum::serve(listener, router).await?;

    Ok(())
}

fn get_detector() -> Box<dyn FaceDetector> {
    FaceDetectorBuilder::new(BlazeFace640(BlazeFaceParams {
        score_threshold: 0.15,
        nms: Nms::default(),
        target_size: 320,
        prior_boxes: PriorBoxesParams::default(),
    }))
        .from_file("blazefaces-640.onnx".to_string())
        .build().unwrap()
}

async fn get_face_pic(
    State(app_state): State<AppState>,
    Path(fid): Path<u32>,
) -> Result<impl IntoResponse, StatusCode> {
    let face_detector = app_state.face_detector.lock().await;
    let mut client = app_state.client.lock().await;

    let filename = format!("{}_noggle.jpg", fid);
    let mut headers = HeaderMap::new();

    headers.insert(header::CONTENT_TYPE, "image/jpeg".parse().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?);
    headers.insert(header::CONTENT_DISPOSITION, format!("attachment; filename=\"{}\"", filename).parse().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?);

    let pfp_bytes = download_pfp(&mut client, fid).await.map_err(|_| StatusCode::NOT_FOUND)?;
    let mut image = image::load_from_memory(pfp_bytes.as_slice()).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let array3 = image.clone()
        .into_rgb8()
        .into_array3();
    let faces = face_detector.detect(array3.view().into_dyn()).unwrap();


    println!("has faces? {}", !faces.is_empty());

    for face in faces.clone() {
        if let Some(landmarks) = face.landmarks {
            if landmarks.len() >= 2 {
                println!("conf: {}, landmarks: {:?}", face.confidence, &landmarks);
                let (l_x, l_y) = landmarks[0];
                let (r_x, r_y) = landmarks[1];
                let dx = r_x - l_x;
                let dy = r_y - l_y;
                let ang = (dy / dx).atan();
                let (mid_x, mid_y) = (l_x + (dx / 2.), l_y + (dy / 2.));
                let len = (dx * dx + dy * dy).sqrt();

                println!("rotation is {}, len is {len}", ang);

                let scale = len / 140.;

                let goggles = get_random_nog(fid).resize_exact((320. * scale) as u32, (120. * scale) as u32, FilterType::Gaussian);

                let x_point = l_x - (120. * scale);
                let y_point = l_y - (60. * scale);

                image::imageops::overlay(&mut image, &goggles, x_point as i64, y_point as i64);
            }
        }
    }
    if !faces.is_empty() {
        let mut vec = Vec::new();
        let mut cursor = Cursor::new(&mut vec);
        image.write_to(&mut cursor, ImageOutputFormat::Jpeg(90)).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let body = Body::from(vec);
        return Ok((StatusCode::OK, headers, body));
    }
    Ok((StatusCode::NOT_FOUND, HeaderMap::default(), Body::empty()))
}

async fn download_pfp(client: &mut HubService, fid: u32) -> Result<Vec<u8>> {

    let profile_pic = client.get_user_profile(fid as u64).await?.profile_picture;

    let bytes = match profile_pic {
        Some(url) => reqwest::get(url).await?.bytes().await?,
        _ => bail!("No profile pic for user")
    };
    Ok(bytes.to_vec())
}

fn get_random_nog(fid: u32) -> DynamicImage {
    let all_nogs = [
        "gogs/black320px.png",
        "gogs/blue320px.png",
        "gogs/blue-med-saturated320px.png",
        "gogs/frog-green320px.png",
        "gogs/green-blue-multi320px.png",
        "gogs/grey-light320px.png",
        "gogs/guava320px.png",
        "gogs/hip-rose320px.png",
        "gogs/honey320px.png",
        "gogs/magenta320px.png",
        "gogs/orange320px.png",
        "gogs/pink-purple-multi320px.png",
        "gogs/red320px.png",
        "gogs/smoke320px.png",
        "gogs/teal320px.png",
        "gogs/watermelon320px.png",
        "gogs/yellow-orange-multi320px.png",
        "gogs/yellow-saturated320px.png",
    ];
    let mut rng = ChaCha8Rng::seed_from_u64(fid as u64);
    let random_selection = rand::seq::index::sample(&mut rng, all_nogs.len(), 1);
    let index = random_selection.index(0);
    image::open(all_nogs.get(index).unwrap()).unwrap()
}


// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Something went wrong: {}", self.0),
        )
            .into_response()
    }
}

// Make our own error that wraps `anyhow::Error`.
struct AppError(eyre::Error);

// This enables using `?` on functions that return `Result<_, anyhow::Error>` to turn them into
// `Result<_, AppError>`. That way you don't need to do that manually.
impl<E> From<E> for AppError
    where
        E: Into<eyre::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}
